import json
import os
from typing import Any, Union
import tiktoken
from pydantic import BaseModel
from .schemas import SummaryResponse, MCQList, RevisionPlan

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.isfile(filepath)

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """Accurately calculates token count using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def format_tokens(text: str) -> int:
    """Legacy helper maintained for backward compatibility."""
    return count_tokens(text)

def export_to_markdown(data: Union[SummaryResponse, MCQList, RevisionPlan]) -> str:
    """Exports structured responses to a clean, human-readable Markdown format."""
    lines = []

    if isinstance(data, SummaryResponse):
        lines.append("# Topic-Wise Summary\n")
        for topic in data.topics:
            lines.append(f"## {topic.topic}")
            lines.append(f"{topic.summary}\n")

    elif isinstance(data, MCQList):
        lines.append(f"# Practice MCQs ({len(data.questions)} Questions)\n")
        for i, q in enumerate(data.questions, 1):
            lines.append(f"### Q{i}: {q.question}\n")
            for idx, opt in enumerate(q.options):
                lines.append(f"- **{chr(65 + idx)}**: {opt}")
            lines.append(f"\n> **Correct Answer:** {q.correct_answer}")
            lines.append(f"> **Explanation:** {q.explanation}\n")

    elif isinstance(data, RevisionPlan):
        lines.append("# 7-Day Revision Plan\n")
        for day in data.plan:
            lines.append(f"## Day {day.day}")
            lines.append(f"- **Estimated Time:** {day.estimated_time_hours} hours")
            lines.append(f"- **Topics:** {', '.join(day.topics)}")
            lines.append("- **Activities:**")
            for act in day.activities:
                lines.append(f"  - {act}")
            lines.append("")
    else:
        lines.append(str(data))

    return "\n".join(lines).strip()

def export_to_json(data: BaseModel, indent: int = 2) -> str:
    """Serializes a Pydantic model to a formatted JSON string."""
    return data.model_dump_json(indent=indent)

def save_output_to_file(content: str, output_path: str):
    """Saves string output to the designated output path, creating directories as needed."""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
