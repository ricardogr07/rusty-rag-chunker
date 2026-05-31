"""End-to-end tests. Require a running Qdrant instance (docker compose up -d).

Run with:
    pytest tests/test_e2e.py -v -m e2e
"""

import pytest
from qdrant_client import QdrantClient

from rusty_rag.config import AppConfig
from rusty_rag.embeddings import embed_texts
from rusty_rag.ingest import ingest_documents
from rusty_rag.retrieve import retrieve
from rusty_rag.vector_store import create_collection


@pytest.fixture(scope="module")
def qdrant_client():
    try:
        client = QdrantClient("localhost", port=6333)
        client.get_collections()
        return client
    except Exception:
        pytest.skip("Qdrant not available — start with: docker compose up -d")


@pytest.fixture(scope="module")
def ingested_collection(qdrant_client: QdrantClient):
    config = AppConfig()
    if qdrant_client.collection_exists(config.collection_name):
        qdrant_client.delete_collection(config.collection_name)
    create_collection(qdrant_client, config)
    ingest_documents("data/raw", config, qdrant_client)
    return config


@pytest.mark.e2e
def test_search_returns_rust_chunk_for_rust_query(
    qdrant_client: QdrantClient, ingested_collection: AppConfig
) -> None:
    config = ingested_collection
    query_vector = embed_texts(["What language handles chunking?"], config)[0]
    results = retrieve(qdrant_client, query_vector, config)

    assert len(results) > 0, "Expected at least one result"
    top = results[0]
    assert top["source_path"].replace("\\", "/").endswith("rust.txt"), (
        f"Expected top result from rust.txt, got: {top['source_path']}"
    )
    assert "Rust" in top["text"], "Expected 'Rust' in top result text"


@pytest.mark.e2e
def test_search_returns_python_chunk_for_embedding_query(
    qdrant_client: QdrantClient, ingested_collection: AppConfig
) -> None:
    config = ingested_collection
    query_vector = embed_texts(["How are embeddings generated?"], config)[0]
    results = retrieve(qdrant_client, query_vector, config)

    assert len(results) > 0, "Expected at least one result"
    sources = [r["source_path"].replace("\\", "/") for r in results]
    assert any(s.endswith("python.txt") for s in sources), (
        f"Expected at least one result from python.txt; got sources: {sources}"
    )
