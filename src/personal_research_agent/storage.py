from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import voyageai
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row

from .config import get_postgres_uri


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = voyageai.Client()
    result = client.embed(texts, model="voyage-4-large")
    return result.embeddings


@contextmanager
def open_postgres_runtime() -> Iterator[tuple[Any, Any]]:
    """Open LangGraph checkpointer and store backed by PostgreSQL."""
    with PostgresSaver.from_conn_string(get_postgres_uri()) as checkpointer:
        checkpointer.setup()
        with PostgresStore.from_conn_string(
            get_postgres_uri(),
            index={
                "dims": 1024,
                "embed": embed_texts,
                "fields": ["text"],
            },
        ) as store:
            store.setup()
            yield checkpointer, store


def delete_thread_checkpoints(thread_id: str) -> None:
    """Delete all LangGraph checkpoints for a session/thread id."""
    with PostgresSaver.from_conn_string(get_postgres_uri()) as checkpointer:
        checkpointer.setup()
        checkpointer.delete_thread(thread_id)


def list_checkpoint_thread_ids() -> list[str]:
    """Return all thread ids currently present in the checkpoint table."""
    with PostgresSaver.from_conn_string(get_postgres_uri()) as checkpointer:
        checkpointer.setup()
    with psycopg.Connection.connect(get_postgres_uri(), autocommit=True, row_factory=dict_row) as conn:
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id").fetchall()
    return [str(row["thread_id"]) for row in rows if row.get("thread_id")]


def prune_orphan_checkpoints(live_thread_ids: set[str]) -> dict[str, Any]:
    """Delete checkpoint threads that no longer have a local session owner."""
    live = {str(item) for item in live_thread_ids if str(item)}
    existing = list_checkpoint_thread_ids()
    orphan_thread_ids = [thread_id for thread_id in existing if thread_id not in live]
    for thread_id in orphan_thread_ids:
        delete_thread_checkpoints(thread_id)
    return {
        "existing_count": len(existing),
        "live_count": len(live),
        "deleted_count": len(orphan_thread_ids),
        "deleted_thread_ids": orphan_thread_ids,
    }
