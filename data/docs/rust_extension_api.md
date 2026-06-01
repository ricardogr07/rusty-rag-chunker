rusty-rag-chunker Rust Extension API

The Rust extension module is named rusty_rag_chunker. It is built with PyO3 and Maturin
and exposes exactly five functions to Python.

Function 1: hello()
Returns a greeting string. Used as a smoke test to verify the Rust extension loaded correctly.

Function 2: count_tokens(text, encoding)
Counts the number of BPE tokens in a string using the specified tiktoken encoding.
The default encoding is cl100k_base, which is the GPT-4 tokenizer.

Function 3: chunk_text(text, max_tokens, overlap_tokens, encoding)
Chunks a single document into a list of dicts, each with a text field.
Every chunk is guaranteed to never exceed max_tokens tokens.
The overlap_tokens parameter controls how many tokens from the end of one chunk
are repeated at the start of the next to preserve context across boundaries.

Function 4: chunk_documents(docs, max_tokens, overlap_tokens, encoding)
Sequential batch chunking across a list of documents.
Amortizes the BPE object initialization cost across the whole batch,
making it significantly faster than calling chunk_text once per document.
Returns a flat list of chunk dicts with text, source_path, and chunk_index fields.

Function 5: chunk_documents_parallel(docs, max_tokens, overlap_tokens, encoding)
Rayon-based parallel batch chunking. Uses all available CPU cores.
Releases the Python GIL for the duration of the chunking work.
On a 50 MB dataset achieves 2.00 MB/s, roughly 40% faster than Python tiktoken.
The crossover point versus Python is between 10 MB and 50 MB of input data.

All five functions use cl100k_base as the default encoding if none is specified.
All token-aware variants produce zero token-limit violations.
The naive Python character-count splitter produces thousands of violations.
