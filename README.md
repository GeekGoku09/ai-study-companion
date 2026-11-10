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
3. **Generates** structured study aids (Summaries, MCQs, Plans) using a centralized LLM interface with explicit prompt templates and structured Pydantic models.

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
    K --> L[OpenAI GPT-4o-mini / GPT-4o]
    E -.->|Retrieve Context| H & I & J
    L --> M[Structured Output: Markdown or JSON]
```

## How to Run / Demo

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Copy `.env.example` to `.env` and set your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```
Or export it directly:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Usage & CLI Options
Run the CLI for different study modes:

#### Topic-wise Summary
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode summary
```

#### Exam-style MCQs
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode mcq --num-mcq 10
```

#### 7-Day Revision Plan
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode plan
```

#### Advanced Options: JSON Output, File Export, Custom Model & Timeout
```bash
# Export summary to JSON file
python3 -m src.main --pdf path/to/notes.pdf --mode summary --format json --output output/summary.json

# Export MCQs to Markdown with custom model and timeout
python3 -m src.main --pdf path/to/notes.pdf --mode mcq --num-mcq 5 --format markdown --output output/mcqs.md --model gpt-4o --timeout 45.0
```

#### CLI Parameters
| Flag | Description | Default | Choices |
| :--- | :--- | :--- | :--- |
| `--pdf` | **Required.** Path to engineering PDF file. | None | File path |
| `--mode` | **Required.** Study companion operation mode. | None | `summary`, `mcq`, `plan` |
| `--num-mcq` | Number of MCQs to generate (for `mcq` mode). | `5` | Integer |
| `--format` | Output format. | `markdown` | `markdown`, `json` |
| `--output` | Destination file path to save output. | `None` (stdout only) | File path |
| `--model` | OpenAI chat completion model name. | `gpt-4o-mini` | String |
| `--timeout`| Timeout in seconds for API network requests. | `None` | Float |

### 4. Running Tests & Quality Checks

Run test suite:
```bash
python3 -m pytest
```

Run static analysis & linting:
```bash
ruff check .
```

## Notes on AI Usage
This project was developed with AI assistance, following a technical-first approach to architecture and logic. For a detailed breakdown of AI vs. human-guided components, see [AI.md](./AI.md).
