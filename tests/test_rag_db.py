import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.rag_db import RagChunk, chunk_markdown_document, rank_rag_records, upsert_rag_chunks


class _FakeCursor:
    def __init__(self):
        self.sql: list[str] = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.sql.append(str(sql))

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass

    def rollback(self):
        pass


class RagDbTests(unittest.TestCase):
    def test_chunk_generation_keeps_required_fields(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "school" / "note.md"
            path.parent.mkdir()
            path.write_text("# 复试安排\n\n2026-05-30 复试材料提交。", encoding="utf-8")

            chunks = chunk_markdown_document(path, root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].path, "school/note.md")
        self.assertEqual(chunks[0].heading, "复试安排")
        self.assertIn("2026-05-30", chunks[0].content)
        self.assertEqual(chunks[0].chunk_id, "school/note.md::chunk-0000")
        self.assertIn("content_hash", chunks[0].metadata)

    def test_upsert_uses_chunk_id_conflict_to_avoid_duplicates(self):
        fake_conn = _FakeConn()
        chunk = RagChunk(
            chunk_id="a.md::chunk-0000",
            path="a.md",
            heading="",
            content="hello",
            metadata={"content_hash": "x"},
        )
        with patch("personal_research_agent.rag_db.psycopg.connect", return_value=fake_conn):
            stats = upsert_rag_chunks([chunk], [[0.1, 0.2, 0.3]], conninfo="postgresql://test", prune_stale=False)

        all_sql = "\n".join(fake_conn.cursor_obj.sql)
        self.assertIn("ON CONFLICT (chunk_id) DO UPDATE", all_sql)
        self.assertEqual(stats["chunks"], 1)

    def test_rank_returns_keyword_date_and_file_name_match(self):
        records = [
            {
                "chunk_id": "2026-05-30.md::chunk-0000",
                "path": "2026-05-30.md",
                "heading": "昆明理工大学复试",
                "content": "昆明理工大学 复试 总分排名 2026-05-30",
                "metadata": {},
                "embedding": [1.0, 0.0, 0.0],
                "updated_at": datetime.now(UTC),
            },
            {
                "chunk_id": "other.md::chunk-0000",
                "path": "other.md",
                "heading": "随笔",
                "content": "天气很好，记录日常安排。",
                "metadata": {},
                "embedding": [0.0, 1.0, 0.0],
                "updated_at": datetime.now(UTC),
            },
        ]

        results = rank_rag_records("2026-05-30 昆明理工 复试", records, [1.0, 0.0, 0.0], top_k=1)

        self.assertEqual(results[0]["chunk_id"], "2026-05-30.md::chunk-0000")
        self.assertGreater(results[0]["final_score"], results[0]["bm25_score"] * 0)


if __name__ == "__main__":
    unittest.main()
