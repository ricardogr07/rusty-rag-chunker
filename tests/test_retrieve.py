"""Integration tests for the retrieve layer. No Docker required."""

import pytest
from qdrant_client import QdrantClient

from rusty_rag.config import AppConfig
from rusty_rag.retrieve import retrieve
from rusty_rag.vector_store import create_collection, upsert_chunks

FAKE_DIM = 4
NUM_DOCS = 5


@pytest.fixture
def cfg() -> AppConfig:
    config = AppConfig()
    config.embedding_dimension = FAKE_DIM
    config.retrieval_top_k = 3
    return config


@pytest.fixture
def populated_client(cfg: AppConfig) -> QdrantClient:
    client = QdrantClient(":memory:")
    create_collection(client, cfg)

    chunks = [
        {"document_id": f"doc_{i:03d}", "source_path": f"doc_{i:03d}.txt",
         "chunk_index": 0, "text": f"chunk text {i}", "token_count": 3}
        for i in range(NUM_DOCS)
    ]
    vectors = [[float(i), 0.0, 0.0, 0.0] for i in range(NUM_DOCS)]
    upsert_chunks(client, chunks, vectors, cfg)
    return client


def test_retrieve_returns_top_k(populated_client: QdrantClient, cfg: AppConfig) -> None:
    results = retrieve(populated_client, [4.0, 0.0, 0.0, 0.0], cfg)
    assert len(results) <= cfg.retrieval_top_k
    assert all("text" in c for c in results)


def test_retrieve_scores_are_present(populated_client: QdrantClient, cfg: AppConfig) -> None:
    results = retrieve(populated_client, [4.0, 0.0, 0.0, 0.0], cfg)
    assert all("score" in c for c in results)


def test_retrieve_returns_relevant_result(populated_client: QdrantClient, cfg: AppConfig) -> None:
    results = retrieve(populated_client, [4.0, 0.0, 0.0, 0.0], cfg)
    # Vector [4.0,0,0,0] is closest to doc_004 whose vector is [4.0,0,0,0]
    assert results[0]["document_id"] == "doc_004"
