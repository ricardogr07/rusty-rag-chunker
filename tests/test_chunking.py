import pytest

from rusty_rag_chunker import chunk_documents, chunk_text, count_tokens

LONG_TEXT = "word " * 5000


def test_chunk_documents_preserves_document_id():
    docs = [{"document_id": "doc_001", "source_path": "x.txt", "text": LONG_TEXT}]
    chunks = chunk_documents(docs, max_tokens=200, overlap_tokens=50)
    assert all(c["document_id"] == "doc_001" for c in chunks)


def test_chunk_documents_indexes_are_sequential():
    docs = [{"document_id": "doc_001", "source_path": "x.txt", "text": LONG_TEXT}]
    chunks = chunk_documents(docs, max_tokens=200, overlap_tokens=50)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_invalid_overlap_raises():
    with pytest.raises(Exception):
        chunk_text(LONG_TEXT, max_tokens=100, overlap_tokens=100)


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0
