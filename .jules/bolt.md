# Bolt's Journal - Critical Learnings

## 2025-05-18 - Avoid Repeated Tokenization in Chunk Accumulators
**Learning:** In chunking algorithms where consecutive items are merged up to a token limit, repeatedly encoding the growing concatenated text (`encoding.encode(current_chunk_content)`) causes O(N^2) tokenization overhead and repetitive string copying. Moreover, `tiktoken.encode_batch` does not necessarily yield a speedup for lists of small paragraphs due to batching overhead compared to clean sequential encoding of individual chunk contents.
**Action:** Always tokenize candidate items once (`tokens_new = len(encoding.encode(chunk.content))`), maintain a running sum of tokens (`current_tokens += tokens_new`), buffer chunks in a list, and join them only when emitting the merged chunk.

## 2026-11-02 - Cache Query Embeddings & Direct Float32 Allocations in FAISS Retrievers
**Learning:** In RAG systems, repeated query retrieval is a major bottleneck because each call invokes an external embedding API roundtrip (~80-150ms). In Python, using `lru_cache` returning a hashable tuple of float embeddings allows 0ms instantaneous cache hits on repeat queries while remaining lightweight. Additionally, converting Python nested float lists to NumPy with `np.array(embeddings).astype('float32')` creates an intermediate 64-bit float array before downcasting; specifying `dtype=np.float32` directly avoids this intermediate allocation and is ~18% faster on vector matrix builds.
**Action:** Use an LRU cache on query embedding calculations, clear it on index rebuilds, and construct NumPy embedding matrices directly with `dtype=np.float32`.
