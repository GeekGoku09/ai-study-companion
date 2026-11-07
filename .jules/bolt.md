# Bolt's Journal - Critical Learnings

## 2025-05-18 - Avoid Repeated Tokenization in Chunk Accumulators
**Learning:** In chunking algorithms where consecutive items are merged up to a token limit, repeatedly encoding the growing concatenated text (`encoding.encode(current_chunk_content)`) causes O(N^2) tokenization overhead and repetitive string copying. Moreover, `tiktoken.encode_batch` does not necessarily yield a speedup for lists of small paragraphs due to batching overhead compared to clean sequential encoding of individual chunk contents.
**Action:** Always tokenize candidate items once (`tokens_new = len(encoding.encode(chunk.content))`), maintain a running sum of tokens (`current_tokens += tokens_new`), buffer chunks in a list, and join them only when emitting the merged chunk.

## 2026-03-30 - NumPy Dtype Cast Surprises on Nested Python Lists vs 1D/2D Query Vectors
**Learning:** When creating a NumPy float32 array from a nested list of floats, `np.array(large_nested_list, dtype=np.float32)` can surprisingly be 2-3x slower than `np.array(large_nested_list, dtype=np.float64).astype(np.float32)` because NumPy's C layer converts Python floats to native C doubles before downcasting element-by-element during parse. However, for single query vectors `[query_embedding]` (1 x 1536), `np.array([query_embedding], dtype=np.float32)` avoids an extra array allocation and copy, resulting in a ~15% speedup over `np.array([query_embedding]).astype('float32')`.
**Action:** For single query vector conversions passed to FAISS `index.search`, specify `dtype=np.float32` directly at array creation. Always benchmark list-to-array conversions when changing dtypes on nested lists.
