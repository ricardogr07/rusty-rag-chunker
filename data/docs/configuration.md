rusty-rag-chunker Configuration Reference

All configuration is read by the AppConfig dataclass in rusty_rag/config.py.
Values can be set via environment variables or a .env file in the project root.

OPENAI_API_KEY
When set, AppConfig automatically switches to text-embedding-3-small (1536 dimensions)
for embeddings and enables the ask command's LLM call to gpt-4o-mini.
When not set, the default local model sentence-transformers/all-MiniLM-L6-v2 is used
(384 dimensions, no API key required).

QDRANT_COLLECTION
Name of the Qdrant vector collection to use. Default: rusty_rag_chunks.
The Wikipedia corpus demo uses rusty_rag_corpus.
A collection created with one embedding model is incompatible with another
because the vector dimensions differ (384 vs 1536).

QDRANT_HOST
Hostname of the Qdrant server. Default: localhost.

QDRANT_PORT
Port of the Qdrant server. Default: 6333.

RETRIEVAL_MIN_SCORE
Minimum cosine similarity score for the ask command to proceed to the LLM.
This is the first layer of the hallucination guard.
Default: 0.50. Recommended range: 0.40 to 0.65.
Queries whose top retrieved chunk scores below this threshold are blocked immediately
and the system prints "I don't have information about this in the knowledge base."

Embedding Models
Two embedding models are supported:
- sentence-transformers/all-MiniLM-L6-v2: 384 dimensions, runs locally, no API key needed, default
- text-embedding-3-small: 1536 dimensions, requires OPENAI_API_KEY, selected automatically when key is set

Default Collection
The default collection name is rusty_rag_chunks.
The retrieval_top_k default is 5 (number of chunks returned per search).
The chunk_max_tokens default is 800 tokens per chunk.
The chunk_overlap_tokens default is 100 tokens of overlap between adjacent chunks.
The tokenizer_encoding default is cl100k_base (same as GPT-4).
