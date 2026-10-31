# Bolt's Journal - Critical Learnings Only

## 2026-03-31 - Quadratic Tokenization Anti-Pattern in RAG Chunking
**Learning:** Re-encoding growing concatenated strings (`len(encoding.encode(accumulated_text))`) inside chunk-grouping loops causes quadratic $O(N^2)$ tokenization overhead. Additionally, repeated calls to `tiktoken.get_encoding(...)` without module-level caching introduce unnecessary dictionary parsing and lookup latency on cold and repeated ingestions.
**Action:** Always track token counts incrementally (`current_tokens += delimiter_tokens + chunk_tokens`) and cache tiktoken encoding instances using `@functools.lru_cache` when grouping document chunks.
