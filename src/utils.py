import json
import os
from typing import Any

import tiktoken

from .schemas import MCQList, RevisionPlan, SummaryResponse


def check_file_exists(filepath: str) -> bool:
    """Checks if a regular file exists at the given path."""
    return os.path.isfile(filepath)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Counts tokens in a string using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except (ValueError, KeyError):
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def format_tokens(text: str) -> int:
    """Accurately calculates token count using tiktoken (cl100k_base)."""
    return count_tokens(text)


def format_summary_markdown(summary: SummaryResponse) -> str:
    """Formats a SummaryResponse into a readable Markdown document."""
    lines = ["# Topic-Wise Summary\n"]
    if not summary.topics:
        lines.append("_No topics found._\n")
        return "\n".join(lines)

    for item in summary.topics:
        lines.append(f"## {item.topic}\n")
        lines.append(f"{item.summary}\n")
    return "\n".join(lines)


def format_mcq_markdown(mcq_list: MCQList) -> str:
    """Formats an MCQList into a readable Markdown document."""
    lines = [f"# Exam-Style Multiple Choice Questions ({len(mcq_list.questions)})\n"]
    if not mcq_list.questions:
        lines.append("_No questions generated._\n")
        return "\n".join(lines)

    for i, q in enumerate(mcq_list.questions, 1):
        lines.append(f"### Q{i}: {q.question}\n")
        for idx, opt in enumerate(q.options):
            letter = chr(ord("A") + idx)
            lines.append(f"- **{letter})** {opt}")
        lines.append(f"\n**Correct Answer:** {q.correct_answer}")
        lines.append(f"**Explanation:** {q.explanation}\n")
    return "\n".join(lines)


def format_plan_markdown(plan: RevisionPlan) -> str:
    """Formats a RevisionPlan into a readable Markdown document."""
    lines = ["# 7-Day Revision Plan\n"]
    if not plan.plan:
        lines.append("_No plan generated._\n")
        return "\n".join(lines)

    for day_plan in plan.plan:
        lines.append(f"## Day {day_plan.day}")
        lines.append(f"**Topics Covered:** {', '.join(day_plan.topics)}")
        lines.append(f"**Estimated Study Time:** {day_plan.estimated_time_hours} hours\n")
        lines.append("### Activities:")
        for act in day_plan.activities:
            lines.append(f"- {act}")
        lines.append("")
    return "\n".join(lines)


def serialize_model_json(model_instance: Any) -> str:
    """Serializes a Pydantic model to a clean, formatted JSON string."""
    if hasattr(model_instance, "model_dump_json"):
        return model_instance.model_dump_json(indent=2)
    return json.dumps(model_instance, indent=2)
