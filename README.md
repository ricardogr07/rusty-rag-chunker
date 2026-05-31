# rusty-rag-chunker

Token-aware text chunking implemented in Rust, exposed to Python via PyO3. Designed as the
performance-critical layer in a retrieval-augmented generation pipeline.

## What It Is

Large language model embedding APIs have hard token limits. A naive character-count splitter
silently produces chunks that exceed those limits, which causes silent truncation at the API
boundary and degrades retrieval quality. This library chunks text by actual BPE token count
using tiktoken-rs (the same tokenizer as GPT-4), with an overlap window to preserve context
at chunk boundaries.

The Rust extension exposes three functions: `chunk_text` for a single document, `chunk_documents`
for sequential batch processing, and `chunk_documents_parallel` for Rayon-parallelised batches.
Each returned chunk carries its exact token count. No chunk ever exceeds `max_tokens`.

## Architecture

```
Python (orchestration)
  ├── rusty_rag/config.py        AppConfig, env-var overrides
  ├── rusty_rag/documents.py     Load .txt / .md from directory
  ├── rusty_rag/embeddings.py    sentence-transformers (local) or OpenAI opt-in
  ├── rusty_rag/vector_store.py  Qdrant create / upsert / search
  ├── rusty_rag/ingest.py        load → chunk → embed → upsert
  ├── rusty_rag/retrieve.py      embed query → search → return dicts
  ├── rusty_rag/prompt.py        context prompt builder + LLM call
  └── rusty_rag/cli.py           init-db / ingest / search / ask (Typer)

Rust (chunking, via PyO3 + Maturin)
  └── src/lib.rs                 chunk_text, chunk_documents, chunk_documents_parallel
```

## Quick Start

**Requirements:** Python ≥ 3.9, Rust toolchain, Docker (for Qdrant).

```bash
# Install Python deps
pip install ".[rag]"

# Build the Rust extension (release mode)
maturin develop --release

# Run tests (no Docker required)
pytest -v

# Start Qdrant
docker compose up -d

# Index the fixture documents
python -m rusty_rag.cli init-db
python -m rusty_rag.cli ingest data/raw

# Search (no API key needed — uses local sentence-transformers)
python -m rusty_rag.cli search "What language handles chunking?"

# Full RAG answer (requires OPENAI_API_KEY)
python -m rusty_rag.cli ask "What problem does Rust solve in this project?"
```

## CLI Reference

| Command | Description |
|---|---|
| `init-db` | Create the Qdrant collection (idempotent) |
| `ingest <dir>` | Load, chunk, embed, and upsert all `.txt` / `.md` files |
| `search <query>` | Semantic search, returns top-k results with scores |
| `ask <question>` | Full RAG: retrieve context, call gpt-4o-mini, print answer |

`search` works with no API key (local embeddings). `ask` requires `OPENAI_API_KEY`.

## Benchmark

Five chunking variants measured across four synthetic dataset sizes. Datasets are repeated
English prose split into 100-document batches. `max_tokens=800`, `overlap_tokens=100`,
encoding `cl100k_base`. Rust extension built with `maturin develop --release`.

| implementation | dataset | total_docs | total_mb | total_chunks | time_s | MB/s | max_tokens | violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| python_naive | tiny | 101 | 0.0098 | 101 | 2.43 | 0.0040 | 25 | 0 |
| python_tiktoken | tiny | 101 | 0.0098 | 101 | 0.006 | 1.677 | 25 | 0 |
| rust_single | tiny | 101 | 0.0098 | 101 | 54.00 | 0.0002 | 24 | 0 |
| rust_batch | tiny | 101 | 0.0098 | 101 | 0.40 | 0.025 | 24 | 0 |
| rust_batch_parallel | tiny | 101 | 0.0098 | 101 | 10.40 | 0.0009 | 24 | 0 |
| python_naive | small | 101 | 1.00 | 301 | 0.92 | 1.090 | 890 | **200** |
| python_tiktoken | small | 101 | 1.00 | 401 | 0.80 | 1.256 | 800 | 0 |
| rust_single | small | 101 | 1.00 | 401 | 64.69 | 0.016 | 800 | 0 |
| rust_batch | small | 101 | 1.00 | 401 | 1.15 | 0.868 | 800 | 0 |
| rust_batch_parallel | small | 101 | 1.00 | 401 | 10.11 | 0.099 | 800 | 0 |
| python_naive | medium | 101 | 10.00 | 2701 | 5.87 | 1.704 | 891 | **2600** |
| python_tiktoken | medium | 101 | 10.00 | 3401 | 6.75 | 1.482 | 800 | 0 |
| rust_single | medium | 101 | 10.00 | 3401 | 66.52 | 0.150 | 800 | 0 |
| rust_batch | medium | 101 | 10.00 | 3401 | 11.65 | 0.858 | 800 | 0 |
| rust_batch_parallel | medium | 101 | 10.00 | 3401 | 11.40 | 0.877 | 800 | 0 |
| python_naive | large | 100 | 50.00 | 13200 | 32.27 | 1.549 | 891 | **13100** |
| python_tiktoken | large | 100 | 50.00 | 16700 | 35.04 | 1.427 | 800 | 0 |
| rust_single | large | 100 | 50.00 | 16700 | 85.50 | 0.585 | 800 | 0 |
| rust_batch | large | 100 | 50.00 | 16700 | 38.65 | 1.294 | 800 | 0 |
| **rust_batch_parallel** | **large** | **100** | **50.00** | **16700** | **24.99** | **2.001** | **800** | **0** |

### Observations

**Violations:** `python_naive` splits by character count with no tokenizer. It produces 200–13,100
chunks that exceed `max_tokens=800` depending on dataset size. Every token-aware variant
(tiktoken and all Rust variants) produces zero violations.

**rust_batch_parallel on large (50 MB):** 2.00 MB/s vs 1.43 MB/s for Python tiktoken — 40% faster.
Rayon distributes documents across CPU cores and the GIL is released for the duration of Rust
processing, allowing full CPU utilisation.

**rust_batch_parallel on medium (10 MB):** 0.88 MB/s vs 1.48 MB/s for Python tiktoken. The
parallelism benefit does not yet outweigh thread-pool startup and the Python–Rust boundary cost
at this scale. The crossover point is between 10 MB and 50 MB for this hardware.

**rust_single:** Consistently the slowest Rust variant. Each per-document call to `chunk_text`
initialises a fresh BPE object, paying the encoding-setup cost 100 times instead of once.
`chunk_documents` and `chunk_documents_parallel` amortise that cost across the entire batch.

**FFI overhead at small scale (tiny, 0.01 MB):** The Python–Rust boundary cost dominates when
documents are short. `python_tiktoken` processes 0.01 MB in 0.006 s (1.68 MB/s) while
`rust_batch` takes 0.40 s (0.025 MB/s). For production use, batch all documents in a single
call to `chunk_documents_parallel` rather than calling `chunk_text` per document.

## Embedding Models

| Model | Dims | API key | Notes |
|---|---:|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | None | Default; downloaded on first use (~90 MB) |
| `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` | Auto-selected when key is present |

Setting `OPENAI_API_KEY` causes `AppConfig` to switch models automatically. A collection
created with one model is incompatible with the other (different vector dimensions).

## Running Tests

```bash
# Unit + integration tests (no Docker)
pytest -v

# E2E tests (requires Docker Compose)
docker compose up -d
pytest tests/test_e2e.py -v -m e2e
```
