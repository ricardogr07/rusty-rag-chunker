"""Benchmark 5 chunking variants across 4 dataset sizes. Writes benchmarks/results.csv."""

import csv
import pathlib
import time

import tiktoken

MAX_TOKENS = 800
OVERLAP_TOKENS = 100
ENCODING = "cl100k_base"

SIZES = ["tiny", "small", "medium", "large"]
SAMPLE_DIR = pathlib.Path("benchmarks/sample_docs")
OUT_CSV = pathlib.Path("benchmarks/results.csv")

FIELDNAMES = [
    "implementation", "dataset", "total_docs", "total_mb",
    "total_chunks", "time_seconds", "mb_per_second",
    "max_token_count", "violations",
]


# ---------------------------------------------------------------------------
# Variant implementations
# ---------------------------------------------------------------------------

def _load_docs(size_name: str) -> list[dict]:
    size_dir = SAMPLE_DIR / size_name
    docs = []
    for path in sorted(size_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({
            "document_id": path.stem,
            "source_path": str(path),
            "text": text,
        })
    return docs


def _total_mb(docs: list[dict]) -> float:
    return sum(len(d["text"].encode("utf-8")) for d in docs) / (1024 * 1024)


def run_python_naive(docs: list[dict]) -> tuple[list[str], int, int]:
    """Character-based split with no tokenizer (5 chars/token heuristic)."""
    enc = tiktoken.get_encoding(ENCODING)
    max_chars = MAX_TOKENS * 5

    chunks = []
    for doc in docs:
        text = doc["text"]
        i = 0
        while i < len(text):
            chunk = text[i : i + max_chars]
            if chunk:
                chunks.append(chunk)
            i += max_chars

    token_counts = [len(enc.encode(c)) for c in chunks]
    violations = sum(1 for tc in token_counts if tc > MAX_TOKENS)
    max_tc = max(token_counts, default=0)
    return chunks, max_tc, violations


def run_python_tiktoken(docs: list[dict]) -> tuple[list[dict], int, int]:
    enc = tiktoken.get_encoding(ENCODING)
    step = MAX_TOKENS - OVERLAP_TOKENS

    chunks = []
    for doc in docs:
        tokens = enc.encode(doc["text"])
        i = 0
        while i < len(tokens):
            window = tokens[i : i + MAX_TOKENS]
            chunks.append({"text": enc.decode(window), "token_count": len(window)})
            if i + MAX_TOKENS >= len(tokens):
                break
            i += step

    max_tc = max((c["token_count"] for c in chunks), default=0)
    violations = sum(1 for c in chunks if c["token_count"] > MAX_TOKENS)
    return chunks, max_tc, violations


def run_rust_single(docs: list[dict]) -> tuple[list[dict], int, int]:
    from rusty_rag_chunker import chunk_text

    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], MAX_TOKENS, OVERLAP_TOKENS, ENCODING))

    max_tc = max((c["token_count"] for c in chunks), default=0)
    violations = sum(1 for c in chunks if c["token_count"] > MAX_TOKENS)
    return chunks, max_tc, violations


def run_rust_batch(docs: list[dict]) -> tuple[list[dict], int, int]:
    from rusty_rag_chunker import chunk_documents

    chunks = chunk_documents(docs, MAX_TOKENS, OVERLAP_TOKENS, ENCODING)
    max_tc = max((c["token_count"] for c in chunks), default=0)
    violations = sum(1 for c in chunks if c["token_count"] > MAX_TOKENS)
    return chunks, max_tc, violations


def run_rust_batch_parallel(docs: list[dict]) -> tuple[list[dict], int, int]:
    from rusty_rag_chunker import chunk_documents_parallel

    chunks = chunk_documents_parallel(docs, MAX_TOKENS, OVERLAP_TOKENS, ENCODING)
    max_tc = max((c["token_count"] for c in chunks), default=0)
    violations = sum(1 for c in chunks if c["token_count"] > MAX_TOKENS)
    return chunks, max_tc, violations


VARIANTS = {
    "python_naive": run_python_naive,
    "python_tiktoken": run_python_tiktoken,
    "rust_single": run_rust_single,
    "rust_batch": run_rust_batch,
    "rust_batch_parallel": run_rust_batch_parallel,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def benchmark_variant(name: str, fn, docs: list[dict], total_mb: float) -> dict:
    print(f"  Running {name}...", flush=True)
    start = time.perf_counter()
    chunks, max_tc, violations = fn(docs)
    elapsed = time.perf_counter() - start

    return {
        "implementation": name,
        "total_docs": len(docs),
        "total_mb": round(total_mb, 4),
        "total_chunks": len(chunks),
        "time_seconds": round(elapsed, 4),
        "mb_per_second": round(total_mb / elapsed, 4) if elapsed > 0 else 0,
        "max_token_count": max_tc,
        "violations": violations,
    }


def main() -> None:
    if not SAMPLE_DIR.exists():
        print("Sample docs not found. Run: python benchmarks/generate_datasets.py")
        raise SystemExit(1)

    results = []
    for size in SIZES:
        size_dir = SAMPLE_DIR / size
        if not size_dir.exists():
            print(f"  Skipping {size} (not generated)")
            continue

        print(f"\nDataset: {size}")
        docs = _load_docs(size)
        total_mb = _total_mb(docs)

        for variant_name, fn in VARIANTS.items():
            row = benchmark_variant(variant_name, fn, docs, total_mb)
            row["dataset"] = size
            results.append(row)

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in FIELDNAMES})

    print(f"\nResults written to {OUT_CSV}")
    print(f"Total rows: {len(results)} (expected 20 for all 4 sizes × 5 variants)")


if __name__ == "__main__":
    main()
