import os
from typing import Union
import tiktoken
from pydantic import BaseModel
from .schemas import SummaryResponse, MCQList, RevisionPlan

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath)

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Accurately counts the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def format_tokens(text: str) -> int:
    """Maintains backward compatibility for token counting."""
    return count_tokens(text)

def format_summary_markdown(summary: SummaryResponse) -> str:
    """Formats a SummaryResponse as Markdown."""
    lines = ["# Topic-wise Summary\n"]
    for t in summary.topics:
        lines.append(f"## {t.topic}\n")
        lines.append(f"{t.summary}\n")
    return "\n".join(lines).strip()

def format_mcq_markdown(mcqs: MCQList) -> str:
    """Formats an MCQList as Markdown."""
    lines = [f"# Generated MCQs ({len(mcqs.questions)} Questions)\n"]
    for i, q in enumerate(mcqs.questions, 1):
        lines.append(f"### Q{i}: {q.question}\n")
        for idx, opt in enumerate(q.options):
            lines.append(f"- **{chr(65+idx)}**) {opt}")
        lines.append(f"\n- **Correct Answer:** {q.correct_answer}")
        lines.append(f"- **Explanation:** {q.explanation}\n")
    return "\n".join(lines).strip()

def format_plan_markdown(plan: RevisionPlan) -> str:
    """Formats a RevisionPlan as Markdown."""
    lines = ["# 7-Day Revision Plan\n"]
    for day in plan.plan:
        lines.append(f"## Day {day.day}\n")
        lines.append(f"- **Estimated Time:** {day.estimated_time_hours} hours")
        lines.append(f"- **Topics:** {', '.join(day.topics)}")
        lines.append("- **Activities:**")
        for act in day.activities:
            lines.append(f"  - {act}")
        lines.append("")
    return "\n".join(lines).strip()

def format_result(
    result: Union[SummaryResponse, MCQList, RevisionPlan],
    output_format: str = "markdown"
) -> str:
    """Formats any Pydantic result model into markdown or JSON string."""
    if output_format.lower() == "json":
        return result.model_dump_json(indent=2)

    if isinstance(result, SummaryResponse):
        return format_summary_markdown(result)
    elif isinstance(result, MCQList):
        return format_mcq_markdown(result)
    elif isinstance(result, RevisionPlan):
        return format_plan_markdown(result)
    elif isinstance(result, BaseModel):
        return result.model_dump_json(indent=2)
    return str(result)

def save_to_file(filepath: str, content: str):
    """Safely saves text content to a destination file, creating directories if needed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
