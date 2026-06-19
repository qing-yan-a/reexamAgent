from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import voyageai
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

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
