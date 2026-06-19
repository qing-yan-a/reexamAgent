from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import jieba
import psycopg
from psycopg.types.json import Jsonb
from rank_bm25 import BM25Okapi

from .config import PROJECT_ROOT, get_postgres_uri
from .storage import embed_texts


RAG_SOURCE_DIR = PROJECT_ROOT / "test"
RAG_TABLE = "rag_chunks"
EMBEDDING_DIMS = 1024
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 160


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    path: str
    heading: str
    content: str
    metadata: dict[str, Any]


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _split_long_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            breakpoint = max(text.rfind("\n\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
            if breakpoint > start + max_chars // 2:
                end = breakpoint + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_markdown_document(path: Path, root: Path = RAG_SOURCE_DIR) -> list[RagChunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _normalize_text(raw)
    if not text:
        return []

    rel_path = _relative_path(path, root)
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match and current_lines:
            sections.append((current_heading, current_lines))
            current_heading = heading_match.group(2).strip()
            current_lines = [line]
        else:
            if heading_match:
                current_heading = heading_match.group(2).strip()
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[RagChunk] = []
    chunk_index = 0
    for heading, lines in sections:
        section_text = _normalize_text("\n".join(lines))
        for piece in _split_long_text(section_text):
            chunk_id = f"{rel_path}::chunk-{chunk_index:04d}"
            chunks.append(
                RagChunk(
                    chunk_id=chunk_id,
                    path=rel_path,
                    heading=heading,
                    content=piece,
                    metadata={
                        "source_root": RAG_SOURCE_DIR.as_posix(),
                        "chunk_index": chunk_index,
                        "content_hash": _stable_hash(piece),
                    },
                )
            )
            chunk_index += 1
    return chunks


def load_markdown_chunks(source_dir: Path = RAG_SOURCE_DIR) -> list[RagChunk]:
    if not source_dir.exists():
        return []
    chunks: list[RagChunk] = []
    for path in sorted(source_dir.rglob("*.md")):
        chunks.extend(chunk_markdown_document(path, source_dir))
    return chunks


def setup_rag_tables(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RAG_TABLE} (
                chunk_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                heading TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                content_hash TEXT NOT NULL,
                embedding vector({EMBEDDING_DIMS}) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS {RAG_TABLE}_path_idx ON {RAG_TABLE} (path)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {RAG_TABLE}_metadata_idx ON {RAG_TABLE} USING GIN (metadata)")
    conn.commit()
    with conn.cursor() as cur:
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {RAG_TABLE}_embedding_hnsw_idx "
                f"ON {RAG_TABLE} USING hnsw (embedding vector_cosine_ops)"
            )
        except Exception:
            conn.rollback()
            return
    conn.commit()


def _vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.10g}" for value in vector) + "]"


def _parse_vector_text(value: Any) -> list[float]:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(part) for part in text.split(",")]


def upsert_rag_chunks(
    chunks: list[RagChunk],
    embeddings: list[list[float]],
    *,
    conninfo: str | None = None,
    prune_stale: bool = True,
) -> dict[str, int]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks 和 embeddings 数量必须一致")
    conninfo = conninfo or get_postgres_uri()
    with psycopg.connect(conninfo) as conn:
        setup_rag_tables(conn)
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                metadata = dict(chunk.metadata)
                metadata["content_hash"] = _stable_hash(chunk.content)
                cur.execute(
                    f"""
                    INSERT INTO {RAG_TABLE}
                        (chunk_id, path, heading, content, metadata, content_hash, embedding, updated_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s::vector, now())
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        path = EXCLUDED.path,
                        heading = EXCLUDED.heading,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (
                        chunk.chunk_id,
                        chunk.path,
                        chunk.heading,
                        chunk.content,
                        Jsonb(metadata),
                        metadata["content_hash"],
                        _vector_literal(embedding),
                    ),
                )
            deleted = 0
            if prune_stale and chunks:
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                cur.execute(
                    f"DELETE FROM {RAG_TABLE} WHERE metadata->>'source_root' = %s AND NOT (chunk_id = ANY(%s))",
                    (RAG_SOURCE_DIR.as_posix(), chunk_ids),
                )
                deleted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
    return {"chunks": len(chunks), "deleted_stale": deleted}


def rebuild_rag_index(
    source_dir: Path = RAG_SOURCE_DIR,
    *,
    embedder: Callable[[list[str]], list[list[float]]] = embed_texts,
    batch_size: int = 32,
) -> dict[str, Any]:
    chunks = load_markdown_chunks(source_dir)
    if not chunks:
        return {"source_dir": str(source_dir), "documents": 0, "chunks": 0, "message": "没有发现可索引的 Markdown 文件。"}

    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings.extend(embedder([chunk.content for chunk in batch]))

    stats = upsert_rag_chunks(chunks, embeddings)
    documents = len({chunk.path for chunk in chunks})
    return {
        "source_dir": str(source_dir),
        "documents": documents,
        "chunks": stats["chunks"],
        "deleted_stale": stats["deleted_stale"],
        "message": f"已重建数据库 RAG 索引：{documents} 个文档，{stats['chunks']} 个 chunk。",
    }


def load_rag_records(*, conninfo: str | None = None) -> list[dict[str, Any]]:
    conninfo = conninfo or get_postgres_uri()
    with psycopg.connect(conninfo) as conn:
        setup_rag_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, path, heading, content, metadata, embedding::text, updated_at
                FROM {RAG_TABLE}
                ORDER BY updated_at DESC, path ASC, chunk_id ASC
                """
            )
            records = []
            for row in cur.fetchall():
                records.append(
                    {
                        "chunk_id": row[0],
                        "path": row[1],
                        "heading": row[2],
                        "content": row[3],
                        "metadata": row[4] or {},
                        "embedding": _parse_vector_text(row[5]),
                        "updated_at": row[6],
                    }
                )
            return records


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [token.strip() for token in jieba.lcut(lowered) if token.strip()]
    tokens.extend(re.findall(r"[a-zA-Z0-9_\-./]+", lowered))
    return [token for token in tokens if len(token) > 1]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def rank_rag_records(
    query: str,
    records: list[dict[str, Any]],
    query_embedding: list[float],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not records:
        return []

    corpus_texts = [
        "\n".join([str(record.get("path", "")), str(record.get("heading", "")), str(record.get("content", ""))])
        for record in records
    ]
    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    max_bm25 = max((float(score) for score in bm25_scores), default=0.0)
    dense_scores = [_cosine_similarity(query_embedding, list(record.get("embedding") or [])) for record in records]

    query_lower = query.lower()
    ranked: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        dense = float(dense_scores[index])
        bm25_score = float(bm25_scores[index])
        dense_norm = (dense + 1.0) / 2.0
        bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
        path = str(record.get("path", "")).lower()
        heading = str(record.get("heading", "")).lower()
        filename_boost = 0.08 if any(part and part in path for part in re.findall(r"[\w\-.]+", query_lower)) else 0.0
        heading_boost = 0.05 if heading and heading in query_lower else 0.0
        final_score = 0.65 * dense_norm + 0.30 * bm25_norm + filename_boost + heading_boost
        ranked.append(
            {
                "chunk_id": record.get("chunk_id", ""),
                "path": record.get("path", ""),
                "heading": record.get("heading", ""),
                "content": record.get("content", ""),
                "metadata": record.get("metadata", {}),
                "final_score": round(final_score, 6),
                "dense_score": round(dense, 6),
                "bm25_score": round(bm25_score, 6),
            }
        )
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked[: max(1, top_k)]


def retrieve_hybrid_rag(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    records = load_rag_records()
    if not records:
        return []
    query_embedding = embed_texts([query])[0]
    return rank_rag_records(query, records, query_embedding, top_k=top_k)
