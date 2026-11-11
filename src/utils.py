import json
import os
from typing import Any, Optional
import tiktoken


def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath)


def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    Accurately counts the number of tokens in a given text using tiktoken.
    Falls back gracefully to standard cl100k_base encoding if specific model not found.
    """
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def format_tokens(text: str) -> int:
    """Backward compatible token counting function."""
    return count_tokens(text)


def write_output_file(content: str, filepath: str) -> None:
    """Writes textual content to a destination file, creating parent directories if needed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def export_to_json(data: Any, indent: int = 2) -> str:
    """Converts Pydantic model or arbitrary python object to formatted JSON string."""
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=indent, ensure_ascii=False)
    elif hasattr(data, "dict"):
        return json.dumps(data.dict(), indent=indent, ensure_ascii=False)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def export_to_markdown(data: Any, mode: Optional[str] = None) -> str:
    """Converts structured models (SummaryResponse, MCQList, RevisionPlan) to clean Markdown."""
    lines = []

    # Check for SummaryResponse
    if hasattr(data, "topics"):
        lines.append("# Topic-Wise Study Summary\n")
        if not data.topics:
            lines.append("_No topics summarized._\n")
        for t in data.topics:
            lines.append(f"## {t.topic}")
            lines.append(f"{t.summary}\n")
        return "\n".join(lines).strip()

    # Check for MCQList
    if hasattr(data, "questions"):
        lines.append("# Exam Practice Questions (MCQs)\n")
        if not data.questions:
            lines.append("_No questions generated._\n")
        for i, q in enumerate(data.questions, start=1):
            lines.append(f"### Q{i}: {q.question}")
            option_letters = ["A", "B", "C", "D"]
            for idx, opt in enumerate(q.options):
                letter = option_letters[idx] if idx < len(option_letters) else str(idx + 1)
                lines.append(f"- **{letter})** {opt}")
            lines.append(f"\n**Correct Answer:** {q.correct_answer}")
            lines.append(f"**Explanation:** {q.explanation}\n")
        return "\n".join(lines).strip()

    # Check for RevisionPlan
    if hasattr(data, "plan"):
        lines.append("# 7-Day Revision Plan\n")
        if not data.plan:
            lines.append("_No plan entries available._\n")
        for day in data.plan:
            lines.append(f"## Day {day.day} ({day.estimated_time_hours} hrs)")
            if day.topics:
                lines.append(f"**Topics:** {', '.join(day.topics)}")
            if day.activities:
                lines.append("**Activities:**")
                for act in day.activities:
                    lines.append(f"- {act}")
            lines.append("")
        return "\n".join(lines).strip()

    # Fallback to string or json representation
    return str(data)
