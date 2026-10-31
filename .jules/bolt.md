# Performance & Architecture Learnings (Bolt Persona)

## FAISS Vector Index Persistence
- Calculating embeddings for dense engineering documents across multiple runs causes unnecessary API latency and cost.
- Storing the FAISS index (`faiss.write_index`) coupled with a JSON chunk metadata registry eliminates redundant calls and allows sub-second query retrieval.

## Batch Embeddings
- Batching embedding generation requests (chunk size: 64) prevents rate limit exhaustion and reduces network overhead compared to individual requests.

## Tiktoken Tokenizer Caching
- Instantiating tiktoken encodings repeatedly across chunks incurs noticeable CPU overhead. Caching the `Encoding` instance in a global dictionary provides near-instantaneous token checks across all chunking operations.

## Context Window Budgeting
- RAG retrieval top-k concatenation can exceed model context bounds on very large chunks. Pre-truncating concatenated contexts safely ensures the model never throws context length exceptions.
