from __future__ import annotations

from typing import TYPE_CHECKING

from rusty_rag_chunker import chunk_documents
from rusty_rag.documents import load_documents
from rusty_rag.embeddings import embed_texts
from rusty_rag.vector_store import upsert_chunks

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from rusty_rag.config import AppConfig


def ingest_documents(directory: str, config: AppConfig, client: QdrantClient) -> int:
    """Load, chunk, embed, and upsert documents. Returns the number of chunks upserted."""
    docs = load_documents(directory)
    chunks = chunk_documents(
        docs,
        config.chunk_max_tokens,
        config.chunk_overlap_tokens,
        config.tokenizer_encoding,
    )
    vectors = embed_texts([c["text"] for c in chunks], config)
    upsert_chunks(client, chunks, vectors, config)
    return len(chunks)
