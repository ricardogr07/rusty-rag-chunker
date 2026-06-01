from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if TYPE_CHECKING:
    from rusty_rag.config import AppConfig


def create_collection(client: QdrantClient, config: AppConfig) -> None:
    if not client.collection_exists(config.collection_name):
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(size=config.embedding_dimension, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    vectors: list[list[float]],
    config: AppConfig,
) -> None:
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c['document_id']}:{c['chunk_index']}")),
            vector=v,
            payload={
                "document_id": c["document_id"],
                "source_path": c["source_path"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "token_count": c["token_count"],
            },
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=config.collection_name, points=points)


def search(client: QdrantClient, query_vector: list[float], config: AppConfig):
    return client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        limit=config.retrieval_top_k,
    ).points
