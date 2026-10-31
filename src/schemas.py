from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
import json

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as source, page, chunk_id")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation")

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content cannot be empty or pure whitespace.")
        return v

class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str = Field(..., min_length=5, description="The examination question text")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Four distinct options")
    correct_answer: str = Field(..., description="The correct answer text or option label (e.g. 'A')")
    explanation: str = Field(..., min_length=5, description="Explanation for why the answer is correct")

    @field_validator("options")
    @classmethod
    def validate_options(cls, options: List[str]) -> List[str]:
        cleaned = [opt.strip() for opt in options]
        if any(len(opt) == 0 for opt in cleaned):
            raise ValueError("MCQ options cannot be empty strings.")
        return cleaned

    @model_validator(mode="after")
    def validate_correct_answer_matches(self) -> "MCQ":
        ans = self.correct_answer.strip()
        opts = [o.strip() for o in self.options]

        # Check if answer matches option text directly
        if ans in opts:
            return self

        # Check if answer is an option letter (A, B, C, D) or index
        valid_prefixes = ["A", "B", "C", "D"]
        if ans.upper() in valid_prefixes:
            return self

        for idx, prefix in enumerate(valid_prefixes):
            if ans.startswith(f"{prefix})") or ans.startswith(f"{prefix}:") or ans.startswith(f"{prefix}."):
                return self
            # Check if answer matches stripped text after prefix
            if idx < len(opts) and ans.lower() == opts[idx].lower():
                return self

        # Fallback: if substring match or relaxed comparison holds, allow it; otherwise warn/validate
        matched = any(ans.lower() in o.lower() or o.lower() in ans.lower() for o in opts)
        if not matched:
            raise ValueError(
                f"Correct answer '{ans}' does not match any of the provided options: {opts}"
            )
        return self

class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Formats the MCQs as Markdown text."""
        md_lines = ["# Multiple Choice Questions\n"]
        for i, q in enumerate(self.questions, 1):
            md_lines.append(f"### Question {i}")
            md_lines.append(f"{q.question}\n")
            for idx, opt in enumerate(q.options):
                label = chr(65 + idx)
                md_lines.append(f"- **{label})** {opt}")
            md_lines.append(f"\n**Correct Answer:** {q.correct_answer}")
            md_lines.append(f"**Explanation:** {q.explanation}\n")
            md_lines.append("---\n")
        return "\n".join(md_lines).strip()

    def to_json(self, indent: int = 2) -> str:
        """Returns JSON representation."""
        return json.dumps(self.model_dump(), indent=indent)

class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str = Field(..., min_length=1, description="The topic title or concept name")
    summary: str = Field(..., min_length=5, description="Summary details and key formulas")

class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Formats the summary as Markdown text."""
        md_lines = ["# Topic Summaries\n"]
        for topic_obj in self.topics:
            md_lines.append(f"## {topic_obj.topic}\n")
            md_lines.append(f"{topic_obj.summary}\n")
            md_lines.append("---\n")
        return "\n".join(md_lines).strip()

    def to_json(self, indent: int = 2) -> str:
        """Returns JSON representation."""
        return json.dumps(self.model_dump(), indent=indent)

class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int = Field(..., ge=1, le=30, description="Day number in the study schedule")
    topics: List[str] = Field(..., min_length=1, description="List of topics to cover on this day")
    activities: List[str] = Field(default_factory=list, description="Specific revision tasks")
    estimated_time_hours: float = Field(..., gt=0.0, le=24.0, description="Estimated study time in hours")

class RevisionPlan(BaseModel):
    """Represents a revision plan."""
    plan: List[RevisionDay] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Formats the revision plan as Markdown text."""
        md_lines = ["# Revision Plan\n"]
        for day in self.plan:
            md_lines.append(f"## Day {day.day} ({day.estimated_time_hours:.1f} hours)")
            md_lines.append(f"**Topics:** {', '.join(day.topics)}\n")
            if day.activities:
                md_lines.append("**Activities:**")
                for act in day.activities:
                    md_lines.append(f"- {act}")
            md_lines.append("\n---\n")
        return "\n".join(md_lines).strip()

    def to_json(self, indent: int = 2) -> str:
        """Returns JSON representation."""
        return json.dumps(self.model_dump(), indent=indent)
