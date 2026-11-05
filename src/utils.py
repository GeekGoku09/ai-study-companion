import os
import json
import logging
from typing import List, Dict, Any
from .schemas import SummaryResponse, MCQList, RevisionPlan
import tiktoken

logger = logging.getLogger(__name__)

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath)

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Accurately counts tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def format_tokens(text: str) -> int:
    """Returns token count using standard cl100k_base encoding."""
    return count_tokens(text)

def export_to_markdown(data: Any, filepath: str) -> None:
    """Exports structured RAG output (Summary, MCQs, or Plan) to a clean Markdown file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    lines: List[str] = []

    if isinstance(data, SummaryResponse):
        lines.append("# Topic-Wise Study Summary\n")
        for topic in data.topics:
            lines.append(f"## {topic.topic}\n")
            lines.append(f"{topic.summary}\n")

    elif isinstance(data, MCQList):
        lines.append("# Practice Examination MCQs\n")
        for i, q in enumerate(data.questions, start=1):
            lines.append(f"### Question {i}: {q.question}\n")
            for idx, opt in enumerate(q.options):
                opt_letter = chr(65 + idx)
                lines.append(f"- **{opt_letter})** {opt}")
            lines.append(f"\n**Correct Answer:** {q.correct_answer}\n")
            lines.append(f"> **Explanation:** {q.explanation}\n")
            lines.append("---\n")

    elif isinstance(data, RevisionPlan):
        lines.append("# 7-Day Structured Revision Plan\n")
        for day in data.plan:
            lines.append(f"## Day {day.day} ({day.estimated_time_hours} hrs)\n")
            if day.topics:
                lines.append(f"**Topics Covered:** {', '.join(day.topics)}\n")
            lines.append("### Activities:")
            for act in day.activities:
                lines.append(f"- {act}")
            lines.append("\n---\n")

    else:
        # Fallback string representation
        lines.append(str(data))

    content = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def export_to_json(data: Any, filepath: str) -> None:
    """Exports structured Pydantic models to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    elif hasattr(data, "dict"):
        payload = data.dict()
    else:
        payload = data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
