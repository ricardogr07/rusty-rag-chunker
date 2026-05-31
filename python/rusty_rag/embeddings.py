from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rusty_rag.config import AppConfig


def embed_texts(texts: list[str], config: AppConfig) -> list[list[float]]:
    if config.embedding_model.startswith("text-embedding"):
        return _embed_openai(texts, config)
    return _embed_local(texts, config)


def _embed_local(texts: list[str], config: AppConfig) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.embedding_model)
    return model.encode(texts, show_progress_bar=False).tolist()


def _embed_openai(texts: list[str], config: AppConfig) -> list[list[float]]:
    import openai
    response = openai.embeddings.create(model=config.embedding_model, input=texts)
    return [entry.embedding for entry in response.data]
