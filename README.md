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
2. **Retrieves** relevant context using vector embeddings (FAISS).
3. **Generates** structured study aids (Summaries, MCQs, Plans) using a centralized LLM interface with explicit prompt templates and Pydantic response models.

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
    L --> M[Structured Output: Markdown / JSON]
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
# Edit .env and set your OPENAI_API_KEY
```
Or export directly:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Usage
Run the CLI for different study modes:

**Topic-wise Summary:**
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode summary
```

**Exam-style MCQs:**
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode mcq --num-mcq 10
```

**7-Day Revision Plan:**
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode plan
```

**Custom Output File and Formats:**
You can save the output directly to a file (`--output`) and select the format (`--format markdown` or `--format json`):
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode summary --format json --output output/summary.json
```

**Custom Model & Timeout:**
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode mcq --model gpt-4o --timeout 90.0
```

### CLI Options Reference
- `--pdf <path>`: (Required) Path to the engineering PDF file.
- `--mode <summary|mcq|plan>`: (Required) Operation mode.
- `--num-mcq <int>`: Number of MCQs to generate (default: `5`).
- `--output <path>`: Optional file path to save the generated output.
- `--format <markdown|json>`: Output format (default: `markdown`).
- `--model <name>`: OpenAI model to use (default: `gpt-4o-mini`).
- `--timeout <float>`: OpenAI API request timeout in seconds (default: `60.0`).

### 4. Running Tests & Linting
Run the test suite:
```bash
python3 -m pytest
```

Run static code analysis:
```bash
ruff check .
```

## Notes on AI Usage
This project was developed with AI assistance (Jules), following a technical-first approach to architecture and logic. For a detailed breakdown of AI vs. human-guided components, see [AI.md](./AI.md).
