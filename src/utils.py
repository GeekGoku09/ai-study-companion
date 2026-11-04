import os
import json
from typing import Optional, Union
import tiktoken
from .schemas import SummaryResponse, MCQList, RevisionPlan

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath) and os.path.isfile(filepath)

def count_tokens(text: str, model_or_encoding: str = "cl100k_base") -> int:
    """
    Accurately counts the number of tokens in a text string using tiktoken.
    Falls back gracefully if the encoding cannot be loaded.
    """
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding(model_or_encoding)
    except Exception:
        try:
            encoding = tiktoken.encoding_for_model(model_or_encoding)
        except Exception:
            return len(text.split())
    return len(encoding.encode(text))

def format_tokens(text: str) -> int:
    """Convenience alias for count_tokens using standard encoding."""
    return count_tokens(text, "cl100k_base")

def format_summary_markdown(summary: SummaryResponse) -> str:
    """Formats a SummaryResponse object as clean Markdown."""
    lines = ["# Document Summary\n"]
    if not summary.topics:
        lines.append("_No topics found._\n")
        return "\n".join(lines)

    for i, topic in enumerate(summary.topics, start=1):
        lines.append(f"## {i}. {topic.topic}\n")
        lines.append(f"{topic.summary.strip()}\n")
    return "\n".join(lines)

def format_mcq_markdown(mcqs: MCQList) -> str:
    """Formats an MCQList object as clean Markdown."""
    lines = ["# Multiple Choice Questions\n"]
    if not mcqs.questions:
        lines.append("_No questions generated._\n")
        return "\n".join(lines)

    for i, q in enumerate(mcqs.questions, start=1):
        lines.append(f"### Q{i}: {q.question}\n")
        for idx, opt in enumerate(q.options):
            lines.append(f"- **{chr(65 + idx)})** {opt}")
        lines.append(f"\n**Correct Answer:** {q.correct_answer}\n")
        lines.append(f"**Explanation:** {q.explanation}\n")
        lines.append("---\n")
    return "\n".join(lines)

def format_plan_markdown(plan: RevisionPlan) -> str:
    """Formats a RevisionPlan object as clean Markdown."""
    lines = ["# 7-Day Revision Plan\n"]
    if not plan.plan:
        lines.append("_No revision plan generated._\n")
        return "\n".join(lines)

    for day_plan in plan.plan:
        lines.append(f"## Day {day_plan.day} ({day_plan.estimated_time_hours} hrs)\n")
        lines.append("**Topics:**")
        for t in day_plan.topics:
            lines.append(f"- {t}")
        lines.append("\n**Activities:**")
        for act in day_plan.activities:
            lines.append(f"- {act}")
        lines.append("\n---\n")
    return "\n".join(lines)

def export_result(
    result: Union[SummaryResponse, MCQList, RevisionPlan],
    output_path: str,
    output_format: Optional[str] = None
) -> str:
    """
    Exports the generated result (SummaryResponse, MCQList, or RevisionPlan)
    to a file in either JSON or Markdown format.
    """
    if output_format is None:
        ext = os.path.splitext(output_path)[1].lower()
        output_format = "json" if ext == ".json" else "markdown"

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if output_format == "json":
        content = json.dumps(result.model_dump(), indent=2)
    else:
        if isinstance(result, SummaryResponse):
            content = format_summary_markdown(result)
        elif isinstance(result, MCQList):
            content = format_mcq_markdown(result)
        elif isinstance(result, RevisionPlan):
            content = format_plan_markdown(result)
        else:
            content = json.dumps(result.model_dump(), indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
