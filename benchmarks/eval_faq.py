"""
FAQ hallucination benchmark.

Runs every question in faq.json through the full RAG pipeline and reports
whether the system's behaviour matches the expected outcome:
  expected=answer  → the system must return a non-empty answer (not blocked)
  expected=blocked → the system must return the "not in knowledge base" sentinel

Usage:
    python benchmarks/eval_faq.py [--faq PATH] [--top-k N] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

_BLOCKED_MARKER = "I don't have information about this in the knowledge base."


def _run_question(question: str, config, client) -> tuple[str, float]:
    """Return (response_text, top_score)."""
    from rusty_rag.embeddings import embed_texts
    from rusty_rag.prompt import ask_llm, build_context_prompt
    from rusty_rag.retrieve import retrieve

    query_vector = embed_texts([question], config)[0]
    chunks = retrieve(client, query_vector, config)

    top_score = chunks[0]["score"] if chunks else 0.0

    if not chunks or top_score < config.retrieval_min_score:
        return _BLOCKED_MARKER, top_score

    prompt = build_context_prompt(question, chunks)
    answer = ask_llm(prompt, config)
    return answer, top_score


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ hallucination benchmark")
    parser.add_argument(
        "--faq",
        default=str(Path(__file__).parent / "faq.json"),
        help="Path to faq.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Chunks to retrieve per question (default 10 to test semantic coverage)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full answers")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    from qdrant_client import QdrantClient

    from rusty_rag.config import AppConfig

    config = AppConfig()
    config.retrieval_top_k = args.top_k
    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    with open(args.faq) as f:
        questions = json.load(f)

    results: list[dict] = []
    category_counts: dict[str, dict[str, int]] = {}

    print(f"\nFAQ Hallucination Benchmark — {len(questions)} questions")
    print(f"Collection: {config.collection_name}  threshold: {config.retrieval_min_score}")
    print("=" * 72)

    for item in questions:
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        expected = item["expected"]

        print(f"\n[{qid}] ({category})")
        print(f"  Q: {question}")

        response, top_score = _run_question(question, config, client)
        is_blocked = response.startswith(_BLOCKED_MARKER)

        passed = (not is_blocked) if expected == "answer" else is_blocked

        status = "PASS" if passed else "FAIL"
        print(f"  score={top_score:.4f}  {status}  ({'blocked' if is_blocked else 'answered'})")

        if args.verbose and not is_blocked:
            snippet = response[:200].replace("\n", " ")
            print(f"  A: {snippet}{'...' if len(response) > 200 else ''}")

        cat = category_counts.setdefault(category, {"pass": 0, "fail": 0, "total": 0})
        cat["total"] += 1
        cat["pass" if passed else "fail"] += 1

        results.append(
            {
                "id": qid,
                "category": category,
                "expected": expected,
                "passed": passed,
                "top_score": round(top_score, 4),
                "blocked": is_blocked,
            }
        )

    print("\n" + "=" * 72)
    print("Results by category:")
    total_pass = total_fail = 0
    for cat, counts in category_counts.items():
        p, f, t = counts["pass"], counts["fail"], counts["total"]
        total_pass += p
        total_fail += f
        print(f"  {cat:<25} {p}/{t} passed")

    total = total_pass + total_fail
    overall = total_pass / total * 100 if total > 0 else 0.0
    print(f"\nOverall: {total_pass}/{total} passed ({overall:.0f}%)")

    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
