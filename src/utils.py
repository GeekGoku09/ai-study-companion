import json
import os
from typing import Any

import tiktoken
from pydantic import BaseModel


def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath) and os.path.isfile(filepath)


def validate_pdf_path(filepath: str) -> None:
    """Validates that the file exists and has a .pdf extension."""
    if not filepath:
        raise ValueError("PDF filepath cannot be empty.")
    if not filepath.lower().endswith(".pdf"):
        raise ValueError(f"File '{filepath}' is not a valid PDF file (must end in .pdf).")
    if not check_file_exists(filepath):
        raise FileNotFoundError(f"PDF file not found at '{filepath}'.")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Counts the number of tokens in a string using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def format_tokens(text: str) -> int:
    """Counts tokens using default cl100k_base encoding."""
    return count_tokens(text)


def format_output_as_json(data: BaseModel | dict[str, Any] | list[Any]) -> str:
    """Converts a Pydantic model or data structure to formatted JSON."""
    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2)
    return json.dumps(data, indent=2)


def format_output_as_markdown(mode: str, data: BaseModel) -> str:
    """Formats structured outputs into readable Markdown."""
    lines = []
    if mode == "summary":
        lines.append("# Topic-Wise Summary\n")
        topics = getattr(data, "topics", [])
        if not topics:
            lines.append("_No topics found in summary._")
        for topic in topics:
            lines.append(f"## {topic.topic}\n")
            lines.append(f"{topic.summary}\n")
    elif mode == "mcq":
        lines.append("# Exam Practice MCQs\n")
        questions = getattr(data, "questions", [])
        if not questions:
            lines.append("_No MCQs generated._")
        for i, q in enumerate(questions, 1):
            lines.append(f"### Question {i}: {q.question}\n")
            for idx, opt in enumerate(q.options):
                choice = chr(65 + idx)
                lines.append(f"- **{choice})** {opt}")
            lines.append(f"\n**Correct Answer:** {q.correct_answer}\n")
            lines.append(f"**Explanation:** {q.explanation}\n")
    elif mode == "plan":
        lines.append("# 7-Day Revision Plan\n")
        days = getattr(data, "plan", [])
        if not days:
            lines.append("_No revision plan generated._")
        for day in days:
            lines.append(f"## Day {day.day} ({day.estimated_time_hours} hours)\n")
            lines.append("**Topics:**")
            for t in day.topics:
                lines.append(f"- {t}")
            lines.append("\n**Activities:**")
            for act in day.activities:
                lines.append(f"- {act}")
            lines.append("")
    else:
        lines.append(str(data))
    return "\n".join(lines).strip()
