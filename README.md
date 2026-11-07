# AI Study Companion RAG System

An intelligent Retrieval-Augmented Generation (RAG) system designed to transform engineering PDFs into actionable study materials, including topic-wise summaries, exam-style MCQs, and structured revision plans.

## Problem
Engineering students often face "information overload" when reviewing dense textbooks and lecture notes. Manually creating summaries and practice questions is time-consuming, and general-purpose AI often "hallucinates" details not present in the specific course material.

## Constraints & Assumptions
- **No Jupyter Notebooks**: Logic is strictly contained in modular `.py` files for production readiness.
- **Code-First Evaluation**: Focus on clean architecture, readable logic, and high testability.
- **Privacy & Security**: No secrets committed; API keys managed via environment variables.
- **Data Source**: Assumes text-searchable engineering PDFs.
- **Grounding**: All generated content is strictly grounded in the uploaded documents to ensure technical accuracy.

## Proposed Solution
A modular RAG pipeline that:
1. **Ingests** PDFs and performs semantic chunking.
2. **Retrieves** relevant context using vector embeddings (FAISS cosine similarity with L2 normalization).
3. **Generates** structured study aids (Summaries, MCQs, Plans) using a centralized LLM interface with explicit prompt templates.

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
    K --> L[OpenAI Structured Outputs]
    E -.->|Retrieve Context| H & I & J
    L --> M[Markdown / JSON Output]
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
Run the CLI for different modes:

**Topic-wise Summary (Markdown or JSON):**
```bash
# Print to console as Markdown
python -m src.main --pdf path/to/notes.pdf --mode summary

# Export as JSON to a file
python -m src.main --pdf path/to/notes.pdf --mode summary --format json --output output/summary.json
```

**Exam-style MCQs:**
```bash
python -m src.main --pdf path/to/notes.pdf --mode mcq --num-mcq 10 --output output/mcq.md
```

**7-Day Revision Plan:**
```bash
python -m src.main --pdf path/to/notes.pdf --mode plan --output output/plan.md
```

#### CLI Flags
| Flag | Type | Default | Description |
|---|---|---|---|
| `--pdf` | string | *required* | Path to the engineering PDF file |
| `--mode` | `summary` \| `mcq` \| `plan` | *required* | Operation mode |
| `--num-mcq` | integer | `5` | Number of MCQs to generate (mode `mcq`, 1-50) |
| `--output` | string | `None` | Optional file path to write generated output |
| `--format` | `markdown` \| `json` | `markdown` | Output formatting format |
| `--model` | string | `gpt-4o-mini` | OpenAI chat completion model |
| `--timeout` | float | `60.0` | API request timeout in seconds |

### 4. Running Tests & Quality Checks
```bash
# Run unit tests
python -m pytest

# Run static analysis and linting
ruff check .
```

## Notes on AI Usage
This project was developed with AI assistance (Jules), following a technical-first approach to architecture and logic. For a detailed breakdown of AI vs. human-guided components, see [AI.md](./AI.md).
