# AI Study Companion RAG System

An intelligent Retrieval-Augmented Generation (RAG) system designed to transform engineering PDFs into actionable study materials, including topic-wise summaries, exam-style MCQs, and structured revision plans.

## Problem
Engineering students often face "information overload" when reviewing dense textbooks and lecture notes. Manually creating summaries and practice questions is time-consuming, and general-purpose AI often "hallucinates" details not present in the specific course material.

## Constraints & Assumptions
- **No Jupyter Notebooks**: Logic is strictly contained in modular `.py` files for production readiness.
- **Code-First Evaluation**: Focus on clean architecture, readable logic, and testability.
- **Privacy & Security**: No secrets committed; API keys managed via environment variables.
- **Data Source**: Assumes high-quality, text-searchable engineering PDFs.
- **Grounding**: All generated content must be strictly grounded in the uploaded documents to ensure technical accuracy.

## Proposed Solution
A modular RAG pipeline that:
1. **Ingests** PDFs and performs semantic chunking.
2. **Retrieves** relevant context using vector embeddings (FAISS) with chunk persistence and batched embeddings.
3. **Generates** structured study aids (Summaries, MCQs, Plans) using a centralized LLM interface with explicit prompt templates, transient error retry, and structured Pydantic schema validation.

## System Architecture

```mermaid
graph TD
    A[PDF Document] --> B[Ingestor: ingest.py]
    B --> C[Semantic Chunks]
    C --> D[Retriever: retriever.py]
    D --> E[FAISS Vector Store]
    
    F[User CLI: main.py] --> G{Select Mode}
    G -->|summary| H[Summarizer: summarizer.py]
    G -->|mcq| I[MCQ Generator: mcq_generator.py]
    G -->|plan| J[Revision Planner: planner.py]
    
    H & I & J --> K[LLM Service: llm.py]
    K --> L[OpenAI GPT-4o-mini]
    E -.->|Retrieve Context| H & I & J
    L --> M[Structured Output / Markdown / JSON]
```

## How to Run / Demo

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Copy `.env.example` to `.env` or export your OpenAI API key:
```bash
cp .env.example .env
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Usage & CLI Options

The CLI supports the following arguments:
- `--pdf`: Path to the engineering PDF file.
- `--mode`: Operation mode: `summary`, `mcq`, or `plan`.
- `--num-mcq`: Number of MCQs to generate (default: 5).
- `--output`: Path to write the output (`.md` for Markdown, `.json` for JSON).
- `--save-index`: Directory to persist the FAISS index and chunk metadata for fast reuse.
- `--load-index`: Directory to load a pre-computed FAISS index and metadata (skips PDF re-parsing and embedding generation).
- `--verbose`: Enable detailed debug logging.

#### Topic-wise Summary
```bash
python -m src.main --pdf path/to/notes.pdf --mode summary --output summary.md
```

#### Exam-style MCQs
```bash
python -m src.main --pdf path/to/notes.pdf --mode mcq --num-mcq 10 --output mcqs.json
```

#### 7-Day Revision Plan
```bash
python -m src.main --pdf path/to/notes.pdf --mode plan --output revision_plan.md
```

#### Saving & Reusing Vector Indices
Save index to avoid re-embedding:
```bash
python -m src.main --pdf path/to/notes.pdf --mode summary --save-index ./my_index
```

Query later directly using the pre-computed index without uploading/parsing the PDF again:
```bash
python -m src.main --load-index ./my_index --mode mcq --num-mcq 5
```

### 4. Running Tests
```bash
python3 -m pytest
```

## Notes on AI Usage
This project was developed with AI assistance (Jules), following a technical-first approach to architecture and logic. For a detailed breakdown of AI vs. human-guided components, see [AI.md](./AI.md).
