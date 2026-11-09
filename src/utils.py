import json
import os
from typing import Any
import tiktoken
from .schemas import MCQList, RevisionPlan, SummaryResponse


def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return bool(filepath and os.path.exists(filepath))


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Accurately counts the number of tokens in a text string using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def format_tokens(text: str) -> int:
    """Counts tokens using tiktoken (kept for backward compatibility)."""
    return count_tokens(text)


def format_summary_markdown(summary: SummaryResponse) -> str:
    """Renders a SummaryResponse model into a readable Markdown string."""
    lines = ["# Topic-Wise Summary\n"]
    if not summary.topics:
        lines.append("*No topics extracted.*")
        return "\n".join(lines)

    for topic in summary.topics:
        lines.append(f"## {topic.topic}\n")
        lines.append(f"{topic.summary}\n")
    return "\n".join(lines).strip()


def format_mcq_markdown(mcq_list: MCQList) -> str:
    """Renders an MCQList model into a readable Markdown string."""
    lines = [f"# Multiple Choice Questions ({len(mcq_list.questions)})\n"]
    if not mcq_list.questions:
        lines.append("*No questions generated.*")
        return "\n".join(lines)

    for i, q in enumerate(mcq_list.questions, start=1):
        lines.append(f"### Q{i}: {q.question}\n")
        for idx, opt in enumerate(q.options):
            lines.append(f"- **{chr(65 + idx)})** {opt}")
        lines.append(f"\n**Correct Answer:** {q.correct_answer}")
        lines.append(f"**Explanation:** {q.explanation}\n")
    return "\n".join(lines).strip()


def format_plan_markdown(plan: RevisionPlan) -> str:
    """Renders a RevisionPlan model into a readable Markdown string."""
    lines = ["# 7-Day Revision Plan\n"]
    if not plan.plan:
        lines.append("*No revision plan generated.*")
        return "\n".join(lines)

    for day in plan.plan:
        lines.append(f"## Day {day.day}")
        topics_str = ", ".join(day.topics) if day.topics else "None"
        lines.append(f"- **Topics:** {topics_str}")
        lines.append(f"- **Estimated Time:** {day.estimated_time_hours} hours")
        lines.append("- **Activities:**")
        for act in day.activities:
            lines.append(f"  - {act}")
        lines.append("")
    return "\n".join(lines).strip()


def serialize_model_to_json(model: Any, indent: int = 2) -> str:
    """Serializes a Pydantic model to a clean JSON string."""
    if hasattr(model, "model_dump_json"):
        # Pydantic v2
        return json.dumps(json.loads(model.model_dump_json()), indent=indent)
    elif hasattr(model, "json"):
        # Pydantic v1 fallback
        return json.dumps(json.loads(model.json()), indent=indent)
    return json.dumps(model, indent=indent, default=str)
