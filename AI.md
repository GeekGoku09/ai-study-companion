# AI Usage Disclosure

In the spirit of transparency, this document outlines the usage of AI in the development and production hardening of the AI Study Companion RAG system.

## AI-Assisted Files
The following files were generated, expanded, and production-hardened with AI assistance (Jules):
- `src/main.py`: Orchestration logic, CLI arguments (`--output`, `--save-index`, `--load-index`), Markdown/JSON export formatters.
- `src/ingest.py`: PDF validation, error handling, text sanitization, token-aware sub-chunking with overlap.
- `src/retriever.py`: FAISS vector index management, embedding batching, index persistence (`save_index`/`load_index`).
- `src/llm.py`: Centralized LLM interface with structured Pydantic parsing and refusal handling.
- `src/schemas.py`: Data models and field validation (e.g., MCQ answer normalization).
- `src/exceptions.py`: Custom domain exception hierarchy.
- `src/utils.py`: Accurate token counting (`tiktoken`), text sanitization, and structured logger configuration.
- `src/summarizer.py`, `src/mcq_generator.py`, `src/planner.py`: Business logic and prompt orchestration for study aids.
- `tests/`: Comprehensive unit and integration test suites (19 unit/integration tests).
- `pytest.ini` & `.env.example`: Configuration templates and test discovery settings.
- `README.md` & `AI.md`: Documentation, architecture diagrams, and quickstart guides.

## Human-Designed Core Logic
The following aspects represent the core architectural decisions and human-guided design:
- **System Architecture**: Multi-module design separating ingestion, vector storage, generation, and CLI interface.
- **Structured Outputs**: Using Pydantic schemas with strict typing to eliminate hallucinated JSON formatting errors.
- **RAG Grounding**: Strict prompt constraints grounding all generation in retrieved chunks to prevent factual drift.
- **Index Persistence**: Caching embeddings and FAISS indexes on disk to minimize API costs and latency across multiple queries.

## Summary
AI was utilized as a principal software engineer and pair programmer to write robust, tested, and documented code, implement standard vector operations and text chunking routines, and complete a production audit of the repository.
