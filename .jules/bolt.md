# Bolt's Journal - Critical Learnings

## 2025-05-18 - Avoid Repeated Tokenization in Chunk Accumulators
**Learning:** In chunking algorithms where consecutive items are merged up to a token limit, repeatedly encoding the growing concatenated text (`encoding.encode(current_chunk_content)`) causes O(N^2) tokenization overhead and repetitive string copying. Moreover, `tiktoken.encode_batch` does not necessarily yield a speedup for lists of small paragraphs due to batching overhead compared to clean sequential encoding of individual chunk contents.
**Action:** Always tokenize candidate items once (`tokens_new = len(encoding.encode(chunk.content))`), maintain a running sum of tokens (`current_tokens += tokens_new`), buffer chunks in a list, and join them only when emitting the merged chunk.
