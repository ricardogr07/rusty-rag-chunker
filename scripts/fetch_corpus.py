"""
Download Wikipedia articles for the real-corpus RAG demo.

Uses the MediaWiki action API directly (httpx) with a proper User-Agent so
the requests are not rejected. The `wikipedia` pip package omits a compliant
User-Agent and gets silently rate-blocked by Wikimedia.

Usage:
    python scripts/fetch_corpus.py            # fetch all ~30 articles
    python scripts/fetch_corpus.py --limit 10 # smaller subset for testing
"""
import argparse
import os
import re
import sys
import time


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "rusty-rag-chunker/0.3 "
    "(https://github.com/ricardogr07/rusty-rag-chunker; rgr5882@gmail.com)"
)

TOPICS = [
    "Retrieval-augmented generation",
    "Large language model",
    "Transformer (machine learning model)",
    "BERT (language model)",
    "GPT-4",
    "Byte pair encoding",
    "Tokenization (lexical analysis)",
    "Word embedding",
    "Word2vec",
    "Cosine similarity",
    "K-nearest neighbors algorithm",
    "Vector database",
    "Rust (programming language)",
    "Python (programming language)",
    "Just-in-time compilation",
    "Garbage collection (computer science)",
    "Memory safety",
    "Natural language processing",
    "Semantic search",
    "Named entity recognition",
    "Sentiment analysis",
    "Information retrieval",
    "Tf–idf",
    "PageRank",
    "Elasticsearch",
    "Apache Lucene",
    "SIMD",
    "Parallel computing",
    "Foreign function interface",
    "OpenAI",
]


def slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def fetch_article(client, title: str) -> tuple[str, str] | None:
    """Return (resolved_title, plain_text) or None on failure."""
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "explaintext": "1",   # strip wiki markup → plain text
        "exsectionformat": "plain",
    }
    resp = client.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        return None

    text = page.get("extract", "").strip()
    if not text:
        return None

    return page["title"], text


def fetch_all(topics: list[str], output_dir: str) -> None:
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install 'httpx'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    succeeded = 0
    failed = []

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for topic in topics:
            slug = slugify(topic)
            path = os.path.join(output_dir, f"{slug}.txt")
            if os.path.exists(path):
                print(f"  skip (exists): {topic}")
                succeeded += 1
                continue

            try:
                result = fetch_article(client, topic)
                if result is None:
                    raise ValueError("page missing or empty")

                resolved_title, text = result
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"{resolved_title}\n\n{text}\n")
                print(f"  ok: {resolved_title} ({len(text):,} chars)")
                succeeded += 1
            except Exception as exc:
                print(f"  FAIL: {topic} — {exc}")
                failed.append(topic)

            time.sleep(0.3)

    print(f"\nDone: {succeeded} fetched, {len(failed)} failed")
    if failed:
        print("Failed topics:", failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia corpus for RAG demo")
    parser.add_argument("--limit", type=int, default=None, help="Fetch only first N topics")
    parser.add_argument("--output", default="data/corpus", help="Output directory")
    args = parser.parse_args()

    topics = TOPICS[: args.limit] if args.limit else TOPICS
    print(f"Fetching {len(topics)} articles → {args.output}/")
    fetch_all(topics, args.output)


if __name__ == "__main__":
    main()
