"""Integration tests for the Qdrant vector store layer. No Docker required."""

import pytest
from qdrant_client import QdrantClient

from rusty_rag.config import AppConfig
from rusty_rag.vector_store import create_collection, search, upsert_chunks

FAKE_DIM = 4  # tiny dimension for fast in-memory tests


@pytest.fixture
def cfg() -> AppConfig:
    config = AppConfig()
    config.embedding_dimension = FAKE_DIM
    return config


@pytest.fixture
def mem_client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_create_collection(mem_client: QdrantClient, cfg: AppConfig) -> None:
    create_collection(mem_client, cfg)
    assert mem_client.collection_exists(cfg.collection_name)


def test_create_collection_is_idempotent(mem_client: QdrantClient, cfg: AppConfig) -> None:
    create_collection(mem_client, cfg)
    create_collection(mem_client, cfg)  # must not raise
    assert mem_client.collection_exists(cfg.collection_name)


def test_upsert_and_search(mem_client: QdrantClient, cfg: AppConfig) -> None:
    create_collection(mem_client, cfg)

    chunks = [
        {"document_id": "doc_001", "source_path": "doc_001.txt", "chunk_index": 0,
         "text": "hello world", "token_count": 2},
        {"document_id": "doc_002", "source_path": "doc_002.txt", "chunk_index": 0,
         "text": "foo bar baz", "token_count": 3},
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    upsert_chunks(mem_client, chunks, vectors, cfg)

    results = search(mem_client, [1.0, 0.0, 0.0, 0.0], cfg)

    assert len(results) > 0
    assert results[0].payload["document_id"] == "doc_001"
    assert "text" in results[0].payload


def test_ingest_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path, cfg: AppConfig) -> None:
    """Full ingest pipeline with embed_texts replaced by a fake to avoid downloading models."""
    (tmp_path / "sample.txt").write_text("This is a test document. " * 30, encoding="utf-8")

    import rusty_rag.ingest as ingest_mod

    def _fake_embed(texts: list[str], config: AppConfig) -> list[list[float]]:
        return [[0.1] * config.embedding_dimension for _ in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", _fake_embed)

    mem_client = QdrantClient(":memory:")
    create_collection(mem_client, cfg)

    count = ingest_mod.ingest_documents(str(tmp_path), cfg, mem_client)
    assert count > 0
