# rusty-rag-chunker

Token-aware text chunking implemented in Rust, exposed to Python via PyO3. Designed as the
performance-critical layer in a retrieval-augmented generation pipeline.

## What It Is

Large language model embedding APIs have hard token limits. A naive character-count splitter
silently produces chunks that exceed those limits, which causes silent truncation at the API
boundary and degrades retrieval quality. This library chunks text by actual BPE token count
using tiktoken-rs (the same tokenizer as GPT-4), with an overlap window to preserve context
at chunk boundaries.

Every chunk is guaranteed to never exceed `max_tokens`. The default encoding is `cl100k_base`,
which matches both GPT-4 and `text-embedding-3-small`.

## Architecture

```
Python (orchestration)
  ├── rusty_rag/config.py        AppConfig, env-var overrides
  ├── rusty_rag/documents.py     Load .txt / .md from directory
  ├── rusty_rag/embeddings.py    sentence-transformers (local) or OpenAI opt-in
  ├── rusty_rag/vector_store.py  Qdrant create / upsert / search
  ├── rusty_rag/ingest.py        load → chunk → embed → upsert
  ├── rusty_rag/retrieve.py      embed query → search → return dicts with scores
  ├── rusty_rag/prompt.py        context prompt builder + LLM call
  └── rusty_rag/cli.py           init-db / ingest / search / ask (Typer)

Rust (chunking, via PyO3 + Maturin)
  └── src/lib.rs                 hello, count_tokens, chunk_text,
                                 chunk_documents, chunk_documents_parallel
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

# Full RAG answer (requires OPENAI_API_KEY in .env)
python -m rusty_rag.cli ask "What problem does Rust solve in this project?"
```

Copy `.env.example` to `.env` and fill in your OpenAI key before using `ask`.

## CLI Reference

| Command | Description |
|---|---|
| `init-db` | Create the Qdrant collection (idempotent) |
| `ingest <dir>` | Load, chunk, embed, and upsert all `.txt` / `.md` files |
| `search <query>` | Semantic search, returns top-k results with scores |
| `ask <question>` | Full RAG: retrieve context, call gpt-4o-mini, print answer |

`search` works with no API key (local embeddings). `ask` requires `OPENAI_API_KEY`.

## Configuration

All settings are read from environment variables or a `.env` file (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Enables OpenAI embeddings + `ask` command; auto-selects `text-embedding-3-small` |
| `QDRANT_COLLECTION` | `rusty_rag_chunks` | Qdrant collection name |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `RETRIEVAL_MIN_SCORE` | `0.50` | Minimum cosine similarity for `ask` (hallucination guard layer 1) |

## Hallucination Guard

The `ask` command uses a two-layer guard to prevent answers invented from training data.

**Layer 1 — Score threshold:** before calling the LLM, the top retrieved chunk's cosine
similarity is checked against `RETRIEVAL_MIN_SCORE`. If it falls below the threshold, the
command prints a refusal and exits without an API call. Off-topic queries (e.g. "best recipe
for paella") score ~0.12 — well below the default 0.50 threshold.

**Layer 2 — System prompt:** if the score passes, the LLM is called with a strict system-role
instruction: "Answer ONLY using the information provided in the context. Do NOT use outside
knowledge or training data." The system role is far more effective for grounding than a
user-message prefix.

## Embedding Models

| Model | Dims | API key | Notes |
|---|---:|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | None | Default; downloaded on first use (~90 MB) |
| `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` | Auto-selected when key is present |

Setting `OPENAI_API_KEY` causes `AppConfig` to switch models automatically. A collection
created with one model is incompatible with the other (different vector dimensions).

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

## Real Corpus Demo

Ingests ~30 Wikipedia articles (ML, Rust, Python, NLP, vector search topics) and four project
documentation files into Qdrant using OpenAI embeddings, then answers questions via gpt-4o-mini.

The demo runs three sections to exercise the full pipeline:

1. **Wikipedia Q&A** — general questions answered from the Wikipedia corpus
2. **Project docs Q&A** — project-specific questions answered from `data/docs/`
   (project overview, Rust API reference, hallucination guard explainer, configuration reference)
3. **Hallucination guard** — off-topic questions that are blocked by the score threshold
   before reaching the LLM

**Prerequisites:** Docker Qdrant running + `OPENAI_API_KEY` in `.env`.

```bash
docker compose up -d
pip install ".[rag,corpus]"
python scripts/fetch_corpus.py         # download ~30 Wikipedia articles to data/corpus/
python scripts/demo.py                 # ingest on first run, then Q&A
```

The corpus articles are committed in `data/corpus/` — `fetch_corpus.py` is only needed
if you want to re-download or extend the corpus.

Run the FAQ benchmark to get a graded pass/fail report across 13 questions:

```bash
python benchmarks/eval_faq.py          # 13/13 expected; uses top_k=10 for coverage testing
python benchmarks/eval_faq.py --verbose  # print full LLM answers
```

### Sample Output

```
Collection   : rusty_rag_corpus
Embedding    : text-embedding-3-small (1536d)
Qdrant       : localhost:6333
Min score    : 0.4

######################################################################
# Wikipedia Corpus — should answer
######################################################################

──────────────────────────────────────────────────────────────────────
Q1: What is retrieval-augmented generation and how does it work?
[ANSWERED] top_score=0.6695  sources=retrieval_augmented_generation.txt, large_language_model.txt

Retrieval-augmented generation (RAG) is a technique that enhances large language
models by enabling them to retrieve and incorporate external information into their
responses. When a query arrives, a retriever finds the most relevant documents from
a vector store by comparing embeddings; those documents are combined with the query
to augment the prompt, and the LLM generates an answer grounded in the retrieved
context rather than relying solely on its parametric memory.

######################################################################
# Project Docs — should answer
######################################################################

──────────────────────────────────────────────────────────────────────
Q2: Explain how Rust is used with Python in this project.
[ANSWERED] top_score=0.5733  sources=project_overview.md, rust_extension_api.md

The chunking engine is written in Rust and exposed to Python via PyO3 and Maturin.
The Python layer handles document loading, embedding, vector storage in Qdrant,
retrieval, and LLM calls. Rust is used specifically to chunk text by actual BPE
token count using tiktoken-rs, which is the same tokenizer as GPT-4. Rust also
provides Rayon-based parallelism for batch processing, releasing the Python GIL
for the duration of the chunking work.

######################################################################
# Out of Scope — should be blocked
######################################################################

──────────────────────────────────────────────────────────────────────
Q1: What is the best recipe for making paella?
[BLOCKED]  top_score=0.1450  threshold=0.4
I don't have information about this in the knowledge base.
```

## Running Tests

```bash
# Unit + integration tests (no Docker)
pytest -v

# E2E tests (requires Docker Compose)
docker compose up -d
pytest tests/test_e2e.py -v -m e2e
```
