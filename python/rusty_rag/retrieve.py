from __future__ import annotations

from typing import TYPE_CHECKING

from rusty_rag.vector_store import search

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from rusty_rag.config import AppConfig


def retrieve(client: QdrantClient, query_vector: list[float], config: AppConfig) -> list[dict]:
    """Search Qdrant by a pre-computed query vector and return payload dicts with scores."""
    results = search(client, query_vector, config)
    return [{**point.payload, "score": point.score} for point in results]
