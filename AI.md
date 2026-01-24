# AI Usage Disclosure

In the spirit of transparency, this document outlines the usage of AI in the development of the AI Study Companion RAG system.

## AI-Assisted Files
The following files were generated or significantly modified with AI assistance (Jules):
- `src/main.py`: Orchestration logic and CLI structure.
- `src/ingest.py`: Semantic chunking algorithms and PDF integration.
- `src/retriever.py`: FAISS index management and embedding integration.
- `src/llm.py`: Prompt engineering and centralized LLM interface.
- `src/summarizer.py`, `src/mcq_generator.py`, `src/planner.py`: Business logic for specialized outputs.
- `tests/`: All unit test suites.
- `README.md` & `AI.md`: Documentation and architecture descriptions.

## Human-Designed Core Logic
The following aspects of the system represent the core architectural decisions and human-guided design:
- **System Architecture**: The multi-module design separating ingestion, retrieval, and generation.
- **Data Modeling**: The definition of Pydantic schemas in `src/schemas.py` to ensure type safety across the RAG pipeline.
- **RAG Strategy**: The choice of semantic chunking over fixed-size windowing and the grounding of LLM prompts in retrieved context.
- **Prompt Constraints**: The decision to keep prompts explicit, constrained, and separate from business logic.

## Summary
AI was used as a co-pilot to accelerate development, handle boilerplate code, and implement standard algorithms (like FAISS search and PDF parsing), while the overall system architecture and engineering trade-offs were guided by the project objectives.
