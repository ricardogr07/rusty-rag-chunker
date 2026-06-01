"""Typer CLI for the rusty-rag pipeline."""

import os
from typing import Optional

import typer
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

app = typer.Typer(add_completion=False)


def _client(host: str, port: int) -> QdrantClient:
    return QdrantClient(host=host, port=port)


@app.command()
def init_db() -> None:
    """Create the Qdrant collection (idempotent)."""
    from rusty_rag.config import AppConfig
    from rusty_rag.vector_store import create_collection

    config = AppConfig()
    client = _client(config.qdrant_host, config.qdrant_port)
    create_collection(client, config)
    typer.echo(f"Collection '{config.collection_name}' is ready.")


@app.command()
def ingest(directory: str = typer.Argument("data/raw")) -> None:
    """Ingest all .txt/.md documents in DIRECTORY into Qdrant."""
    from rusty_rag.config import AppConfig
    from rusty_rag.ingest import ingest_documents
    from rusty_rag.vector_store import create_collection

    config = AppConfig()
    client = _client(config.qdrant_host, config.qdrant_port)
    create_collection(client, config)

    n = ingest_documents(directory, config, client)
    typer.echo(f"Chunks ingested: {n}")
    typer.echo(f"Collection:      {config.collection_name}")


@app.command()
def search(
    query: str,
    top_k: Optional[int] = typer.Option(None, help="Number of results (default: from config)"),
) -> None:
    """Semantic search (no LLM required)."""
    from rusty_rag.config import AppConfig
    from rusty_rag.embeddings import embed_texts
    from rusty_rag.retrieve import retrieve

    config = AppConfig()
    if top_k is not None:
        config.retrieval_top_k = top_k
    client = _client(config.qdrant_host, config.qdrant_port)

    query_vector = embed_texts([query], config)[0]
    results = retrieve(client, query_vector, config)

    typer.echo(f'\nQuery: "{query}"\n')
    for i, r in enumerate(results, 1):
        snippet = r["text"][:120].replace("\n", " ")
        typer.echo(
            f"[{i}] score={r['score']:.4f}  source={r['source_path']}  chunk={r['chunk_index']}\n"
            f"    {snippet}"
        )


@app.command()
def ask(
    question: str,
    top_k: Optional[int] = typer.Option(None, help="Number of chunks to retrieve (default: from config)"),
) -> None:
    """Full RAG answer (requires OPENAI_API_KEY)."""
    if not os.getenv("OPENAI_API_KEY"):
        typer.echo(
            "Error: OPENAI_API_KEY is not set. The 'ask' command requires an OpenAI key.",
            err=True,
        )
        raise typer.Exit(code=1)

    from rusty_rag.config import AppConfig
    from rusty_rag.embeddings import embed_texts
    from rusty_rag.prompt import ask_llm, build_context_prompt
    from rusty_rag.retrieve import retrieve

    config = AppConfig()
    if top_k is not None:
        config.retrieval_top_k = top_k
    client = _client(config.qdrant_host, config.qdrant_port)

    query_vector = embed_texts([question], config)[0]
    chunks = retrieve(client, query_vector, config)

    if not chunks or chunks[0]["score"] < config.retrieval_min_score:
        best = chunks[0]["score"] if chunks else 0.0
        typer.echo(
            f"I don't have information about this in the knowledge base.\n"
            f"(Best match score: {best:.3f}, threshold: {config.retrieval_min_score})"
        )
        return

    prompt = build_context_prompt(question, chunks)
    answer = ask_llm(prompt, config)
    typer.echo(answer)


if __name__ == "__main__":
    app()
