"""
End-to-end RAG demo using the real Wikipedia corpus + project docs.

Demonstrates three scenarios:
  1. Wikipedia corpus questions  — answered from retrieved context
  2. Project-specific questions  — answered from data/docs/project_overview.md
  3. Out-of-scope questions      — blocked by the score-threshold hallucination guard

Requires:
  - Docker Qdrant running (docker compose up -d)
  - OPENAI_API_KEY set in environment or .env
  - Corpus downloaded: python scripts/fetch_corpus.py

Usage:
    python scripts/demo.py
"""
import os
import sys
from pathlib import Path

# Allow running as a script from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("QDRANT_COLLECTION", "rusty_rag_corpus")

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"

WIKIPEDIA_QUESTIONS = [
    "What is retrieval-augmented generation and how does it work?",
    "How does byte pair encoding tokenization work?",
    "What makes Rust's memory safety model different from garbage collection?",
    "What is cosine similarity and why is it used for vector search?",
    "How do large language models like GPT-4 differ from earlier NLP models?",
]

PROJECT_QUESTIONS = [
    "What problem does Rust solve in this project?",
    "Explain how Rust is used with Python in this project.",
    "What are the five functions exposed by the Rust extension?",
    "How does the hallucination guard work in the ask command?",
]

OUT_OF_SCOPE_QUESTIONS = [
    "What is the best recipe for making paella?",
    "Who won the FIFA World Cup in 2022?",
]


def check_prerequisites() -> list[str]:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your environment or in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    corpus_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    if not corpus_files:
        print(
            f"Error: no .txt files found in {CORPUS_DIR}.\n"
            "Run: python scripts/fetch_corpus.py",
            file=sys.stderr,
        )
        sys.exit(1)

    return corpus_files


def ensure_ingested(client, config, directory: Path, label: str) -> None:
    from rusty_rag.ingest import ingest_documents
    from rusty_rag.vector_store import create_collection

    create_collection(client, config)

    try:
        info = client.get_collection(config.collection_name)
        count = info.points_count
    except Exception:
        count = 0

    if count and count > 0:
        print(f"Collection '{config.collection_name}' already has {count} points — skipping ingest.\n")
        return

    print(f"Ingesting {label} into '{config.collection_name}'...")
    n = ingest_documents(str(directory), config, client)
    print(f"Ingested {n} chunks.\n")


def _ask(question: str, client, config) -> tuple[str, float, list[str]]:
    """Run one question through the full pipeline. Returns (answer, top_score, sources)."""
    from rusty_rag.embeddings import embed_texts
    from rusty_rag.prompt import ask_llm, build_context_prompt
    from rusty_rag.retrieve import retrieve

    query_vector = embed_texts([question], config)[0]
    chunks = retrieve(client, query_vector, config)
    top_score = chunks[0]["score"] if chunks else 0.0

    if not chunks or top_score < config.retrieval_min_score:
        return "I don't have information about this in the knowledge base.", top_score, []

    sources = list({os.path.basename(c["source_path"]) for c in chunks})
    prompt = build_context_prompt(question, chunks)
    answer = ask_llm(prompt, config)
    return answer, top_score, sources


def run_section(title: str, questions: list[str], client, config) -> None:
    print(f"\n{'#' * 70}")
    print(f"# {title}")
    print(f"{'#' * 70}")

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 70}")
        print(f"Q{i}: {question}")
        answer, top_score, sources = _ask(question, client, config)
        blocked = answer.startswith("I don't have information")

        if blocked:
            print(f"[BLOCKED]  top_score={top_score:.4f}  threshold={config.retrieval_min_score}")
            print(answer)
        else:
            print(f"[ANSWERED] top_score={top_score:.4f}  sources={', '.join(sources)}")
            print(f"\n{answer}")


def main() -> None:
    check_prerequisites()

    from qdrant_client import QdrantClient

    from rusty_rag.config import AppConfig

    config = AppConfig()
    print(f"Collection   : {config.collection_name}")
    print(f"Embedding    : {config.embedding_model} ({config.embedding_dimension}d)")
    print(f"Qdrant       : {config.qdrant_host}:{config.qdrant_port}")
    print(f"Min score    : {config.retrieval_min_score}")

    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    # Ingest corpus on first run (idempotent)
    ensure_ingested(client, config, CORPUS_DIR, f"corpus ({CORPUS_DIR})")

    run_section("Wikipedia Corpus — should answer", WIKIPEDIA_QUESTIONS, client, config)
    run_section("Project Docs — should answer", PROJECT_QUESTIONS, client, config)
    run_section("Out of Scope — should be blocked", OUT_OF_SCOPE_QUESTIONS, client, config)

    print(f"\n{'#' * 70}")
    print("Demo complete.")


if __name__ == "__main__":
    main()
