from rusty_rag_chunker import chunk_text

LONG_TEXT = "word " * 5000


def test_chunk_text_never_exceeds_max_tokens():
    chunks = chunk_text(LONG_TEXT, max_tokens=200, overlap_tokens=50)
    assert all(c["token_count"] <= 200 for c in chunks)


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", max_tokens=200, overlap_tokens=50) == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("hello world", max_tokens=200, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
