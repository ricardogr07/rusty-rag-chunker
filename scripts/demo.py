"""
End-to-end RAG demo using the real Wikipedia corpus.

Requires:
  - Docker Qdrant running (docker compose up -d)
  - OPENAI_API_KEY set in environment or .env
  - Corpus downloaded: python scripts/fetch_corpus.py

Usage:
    QDRANT_COLLECTION=rusty_rag_corpus OPENAI_API_KEY=sk-... python scripts/demo.py

The script ingests data/corpus/ on first run (detected by empty collection).
Subsequent runs skip ingest and go straight to Q&A.
"""
import os
import sys

# Add python/ to the path so rusty_rag is importable when running as a script
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "python")))

# Load .env before importing anything that reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default collection for the real corpus so callers don't need to set it manually.
# AppConfig still defaults to rusty_rag_chunks for all other CLI commands.
os.environ.setdefault("QDRANT_COLLECTION", "rusty_rag_corpus")


QUESTIONS = [
    "What is retrieval-augmented generation and how does it work?",
    "How does byte pair encoding tokenization work?",
    "What makes Rust's memory safety model different from garbage collection?",
    "What is cosine similarity and why is it used for vector search?",
    "How do large language models like GPT-4 differ from earlier NLP models?",
    "What are the advantages of vector databases over traditional keyword search?",
    "How does a foreign function interface allow Rust code to be called from Python?",
]

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rusty_rag_corpus")


def check_prerequisites() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your environment or in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    corpus_files = [
        f for f in os.listdir(CORPUS_DIR)
        if f.endswith(".txt") and f != ".gitkeep"
    ]
    if not corpus_files:
        print(
            f"Error: no .txt files found in {CORPUS_DIR}.\n"
            "Run: python scripts/fetch_corpus.py",
            file=sys.stderr,
        )
        sys.exit(1)

    return corpus_files


def ensure_ingested(client, config) -> None:
    from rusty_rag.vector_store import create_collection
    from rusty_rag.ingest import ingest_documents

    create_collection(client, config)

    try:
        info = client.get_collection(config.collection_name)
        count = info.points_count
    except Exception:
        count = 0

    if count and count > 0:
        print(f"Collection '{config.collection_name}' already has {count} points — skipping ingest.\n")
        return

    corpus_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "corpus"))
    print(f"Ingesting {corpus_path} into '{config.collection_name}'...")
    n = ingest_documents(corpus_path, config, client)
    print(f"Ingested {n} chunks.\n")


def run_demo(client, config) -> None:
    from rusty_rag.embeddings import embed_texts
    from rusty_rag.retrieve import retrieve
    from rusty_rag.prompt import build_context_prompt, ask_llm

    for i, question in enumerate(QUESTIONS, 1):
        print(f"{'='*70}")
        print(f"Q{i}: {question}")
        print()

        query_vector = embed_texts([question], config)[0]
        chunks = retrieve(client, query_vector, config)

        sources = list({c["source_path"] for c in chunks})
        print(f"Sources: {', '.join(os.path.basename(s) for s in sources)}")
        print()

        prompt = build_context_prompt(question, chunks)
        answer = ask_llm(prompt, config)
        print(f"Answer:\n{answer}")
        print()


def main() -> None:
    check_prerequisites()

    # Importing here so env vars (QDRANT_COLLECTION, OPENAI_API_KEY) are already set
    from qdrant_client import QdrantClient
    from rusty_rag.config import AppConfig

    config = AppConfig()
    print(f"Collection : {config.collection_name}")
    print(f"Embedding  : {config.embedding_model} ({config.embedding_dimension}d)")
    print(f"Qdrant     : {config.qdrant_host}:{config.qdrant_port}")
    print()

    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    ensure_ingested(client, config)
    run_demo(client, config)


if __name__ == "__main__":
    main()
