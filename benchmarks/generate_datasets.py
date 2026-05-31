"""Generate synthetic benchmark datasets at four sizes."""

import os
import pathlib

PARAGRAPH = "The quick brown fox jumps over the lazy dog. " * 20

SIZES = {
    "tiny":   10 * 1024,
    "small":  1 * 1024 * 1024,
    "medium": 10 * 1024 * 1024,
    "large":  50 * 1024 * 1024,
}

DOCS_PER_BATCH = 100
OUT_DIR = pathlib.Path("benchmarks/sample_docs")


def _generate_text(target_bytes: int) -> str:
    reps = (target_bytes // len(PARAGRAPH)) + 1
    return (PARAGRAPH * reps)[:target_bytes]


def generate(size_name: str, target_bytes: int) -> None:
    size_dir = OUT_DIR / size_name
    size_dir.mkdir(parents=True, exist_ok=True)

    full_text = _generate_text(target_bytes)
    chunk_size = max(1, len(full_text) // DOCS_PER_BATCH)
    docs = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    for idx, doc_text in enumerate(docs):
        path = size_dir / f"doc_{idx:04d}.txt"
        path.write_text(doc_text, encoding="utf-8")

    print(f"  {size_name:8s}  {len(docs)} docs  ({target_bytes / 1024 / 1024:.2f} MB target)")


if __name__ == "__main__":
    print("Generating benchmark datasets...")
    for name, byte_count in SIZES.items():
        generate(name, byte_count)
    print(f"Done. Files written to {OUT_DIR}/")
