import os
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "rusty_rag_chunks")
    )
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100
    tokenizer_encoding: str = "cl100k_base"
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.50

    def __post_init__(self) -> None:
        if os.getenv("OPENAI_API_KEY"):
            self.embedding_model = "text-embedding-3-small"
            self.embedding_dimension = 1536
