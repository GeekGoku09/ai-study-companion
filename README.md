# AI Study Companion RAG System

An intelligent Retrieval-Augmented Generation (RAG) system designed to transform dense engineering PDFs into actionable study materials: topic-wise summaries, exam-style Multiple Choice Questions (MCQs), and structured revision plans.

---

## Key Features & Architecture
- **Semantic & Resilient Chunking**: Paragraph-aware token chunking with O(N) single-pass tokenization and automatic sentence sub-splitting for oversized blocks.
- **FAISS Vector Search with Persistence**: Indexing with OpenAI embeddings (`text-embedding-3-small`), batching support, and `--save-index` / `--load-index` capabilities for zero re-computation on repeated runs.
- **Structured Pydantic Validation**: Guarantees typed, valid schemas for topics, MCQs, and revision plans with strict model validation.
- **Production Resilience**: Exponential backoff retry handler for OpenAI rate limits and network transient errors, prompt injection isolation delimiters (`<<DOCUMENT_CONTEXT>>`), and clean CLI exit codes.
- **Flexible Export**: Direct terminal display and export to formatted Markdown (`.md`) or structured JSON (`.json`) with `--output`.

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
    L --> M[Structured Output / Export]
```

---

## Installation & Setup

### 1. Requirements
- Python 3.10+ (tested on Python 3.12)
- OpenAI API Key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the example environment configuration:
```bash
cp .env.example .env
```
And populate your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Optional configurations:
- `OPENAI_MODEL`: Chat model (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)

---

## CLI Usage

### Basic Usage
The CLI entrypoint is `src.main`. You can run three modes: `summary`, `mcq`, or `plan`.

#### 1. Topic-wise Summary
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode summary
```

#### 2. Exam-style MCQs
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode mcq --num-mcq 10
```

#### 3. 7-Day Revision Plan
```bash
python3 -m src.main --pdf path/to/notes.pdf --mode plan
```

---

### Advanced CLI Flags

| Flag | Description |
|------|-------------|
| `--output <filepath>` | Export results to a file. Format is inferred from extension (`.md` or `.json`). |
| `--save-index <directory>` | Save the computed FAISS index and chunk metadata to disk. |
| `--load-index <directory>` | Load a pre-computed FAISS index, bypassing PDF extraction & embedding. |
| `--verbose` | Enable detailed debug logs (including API retry and retrieval details). |

#### Examples:

**Exporting MCQs to Markdown:**
```bash
python3 -m src.main --pdf notes.pdf --mode mcq --num-mcq 5 --output mcqs.md
```

**Exporting Summary to JSON:**
```bash
python3 -m src.main --pdf notes.pdf --mode summary --output summary.json
```

**Building & Saving Index Once:**
```bash
python3 -m src.main --pdf notes.pdf --mode summary --save-index ./indexes/thermo_idx
```

**Reusing Existing Index (Zero PDF Re-parse / Zero Re-embedding):**
```bash
python3 -m src.main --load-index ./indexes/thermo_idx --mode mcq --num-mcq 10 --output mcqs.md
```

---

## Testing

Run the full pytest suite:
```bash
python3 -m pytest
```

Run test suite with verbose output:
```bash
python3 -m pytest -v
```

---

## Notes on AI Usage
For disclosures and architectural rationale regarding AI-assisted development, see [AI.md](./AI.md).
