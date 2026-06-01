# rusty-rag-chunker — Project Overview

rusty-rag-chunker is a hybrid Python and Rust library for token-aware text chunking in a
retrieval-augmented generation pipeline. The chunking engine is written in Rust and exposed
to Python via PyO3 and Maturin. The Python layer handles document loading, embedding,
vector storage in Qdrant, retrieval, and LLM calls.

## Why Rust is used in this project

Large language model embedding APIs have hard token limits. A naive character-count splitter
silently produces chunks that exceed those limits, causing silent truncation at the API
boundary and degrading retrieval quality. Rust solves this by chunking text by actual BPE
token count using tiktoken-rs, which is the same tokenizer as GPT-4. Rust also provides
Rayon-based parallelism for batch processing, releasing the Python GIL for the duration of
the chunking work.

## Rust extension functions

The Rust extension is named rusty_rag_chunker. It exposes five functions:

- hello() — smoke test, returns a greeting string
- count_tokens(text, encoding) — counts BPE tokens in a string
- chunk_text(text, max_tokens, overlap_tokens, encoding) — chunks a single document
- chunk_documents(docs, max_tokens, overlap_tokens, encoding) — sequential batch chunking
- chunk_documents_parallel(docs, max_tokens, overlap_tokens, encoding) — Rayon parallel batch

Every chunk is guaranteed to never exceed max_tokens. The default encoding is cl100k_base,
which is the same encoding used by GPT-4 and text-embedding-3-small.

## Benchmark results

The benchmark compares five chunking implementations across four dataset sizes using
max_tokens=800, overlap_tokens=100, and cl100k_base encoding.

On a 50 MB dataset (large), rust_batch_parallel achieves 2.00 MB/s versus 1.43 MB/s for
python_tiktoken, a 40% throughput improvement. The crossover point between Rust and Python
performance is between 10 MB and 50 MB for the test hardware. For datasets smaller than
10 MB, Python tiktoken is faster due to the Python-Rust FFI boundary cost.

python_naive (character-based splitting) produces violations — chunks that exceed max_tokens
— on every dataset larger than tiny: 200 violations on small, 2600 on medium, 13100 on large.
All token-aware variants (Python tiktoken and all Rust variants) produce zero violations.

rust_single is the slowest Rust variant because it initializes a fresh BPE object per
document. chunk_documents and chunk_documents_parallel amortize that cost across the batch.

## Python package: rusty_rag

The Python orchestration layer is the rusty_rag package located at rusty_rag/ in the
project root. It contains these modules:

- config.py — AppConfig dataclass with environment variable overrides
- documents.py — loads .txt and .md files from a directory
- embeddings.py — local sentence-transformers or OpenAI text-embedding-3-small
- vector_store.py — Qdrant collection create, upsert, and search operations
- ingest.py — full pipeline: load, chunk (Rust), embed, upsert
- retrieve.py — embed query, search Qdrant, return payload dicts with scores
- prompt.py — builds the context prompt and calls gpt-4o-mini
- cli.py — Typer CLI with four commands

## CLI commands

The CLI is run as: python -m rusty_rag.cli

The four commands are:

- init-db — creates the Qdrant collection, idempotent
- ingest <directory> — loads all .txt and .md files, chunks with Rust, embeds, upserts
- search <query> — semantic search, returns top-k results with scores, no API key needed
- ask <question> — full RAG: embed query, retrieve chunks, build prompt, call gpt-4o-mini

The search command works without an OpenAI API key using local sentence-transformers.
The ask command requires OPENAI_API_KEY to be set.

## Embedding models

Two embedding models are supported:

- sentence-transformers/all-MiniLM-L6-v2 — 384 dimensions, runs locally, no API key, default
- text-embedding-3-small — 1536 dimensions, requires OPENAI_API_KEY, auto-selected when key is set

Setting OPENAI_API_KEY causes AppConfig to automatically switch to text-embedding-3-small.
A Qdrant collection created with one model is incompatible with the other because the
vector dimensions differ. The default collection name is rusty_rag_chunks. The Wikipedia
corpus demo uses rusty_rag_corpus with OpenAI embeddings.

## Configuration via environment variables

AppConfig reads the following environment variables from a .env file or the shell:

- OPENAI_API_KEY — enables OpenAI embeddings and the ask command LLM call
- QDRANT_COLLECTION — collection name (default: rusty_rag_chunks)
- QDRANT_HOST — Qdrant host (default: localhost)
- QDRANT_PORT — Qdrant port (default: 6333)
- RETRIEVAL_MIN_SCORE — minimum cosine similarity for the ask command (default: 0.50)

## Hallucination guard

The ask command has a two-layer hallucination guard. Layer one is a score threshold: if the
top retrieved chunk's cosine similarity is below RETRIEVAL_MIN_SCORE, the command prints
"I don't have information about this in the knowledge base" and exits without calling the LLM.
Layer two is a system prompt that instructs gpt-4o-mini to answer only from the provided
context and refuse to use outside knowledge.

## Running tests

Unit and integration tests run without Docker: pytest -v
End-to-end tests require Docker Compose: docker compose up -d && pytest tests/test_e2e.py -v -m e2e

## Quick start

Requirements are Python 3.9 or newer, a Rust toolchain, and Docker for Qdrant.

Step 1: pip install ".[rag]"
Step 2: maturin develop --release
Step 3: docker compose up -d
Step 4: python -m rusty_rag.cli init-db
Step 5: python -m rusty_rag.cli ingest data/raw
Step 6: python -m rusty_rag.cli search "What language handles chunking?"
Step 7: python -m rusty_rag.cli ask "What problem does Rust solve in this project?"
