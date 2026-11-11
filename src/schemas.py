from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_answer: str
    explanation: str

    @model_validator(mode="after")
    def validate_correct_answer(self) -> "MCQ":
        # Allow answer either by matching text or letter option (e.g. "A", "B", "C", "D")
        option_letters = ["A", "B", "C", "D"]
        valid_options = set(self.options)
        valid_keys = set(option_letters[:len(self.options)])

        ans = self.correct_answer.strip()
        if ans not in valid_options and ans.upper() not in valid_keys:
            # Check if answer starts with "A) ... "
            matching_prefix = [opt for opt in self.options if opt.strip() == ans or f"{ans})".lower() in opt.lower()]
            if not matching_prefix:
                raise ValueError(
                    f"correct_answer '{self.correct_answer}' must match one of the options {self.options} "
                    f"or an option key {option_letters[:len(self.options)]}."
                )
        return self


class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ]


class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str
    summary: str


class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary]


class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int = Field(..., ge=1, le=30, description="Day index in the revision plan")
    topics: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    estimated_time_hours: float = Field(..., ge=0.1, le=24.0, description="Estimated study time in hours")


class RevisionPlan(BaseModel):
    """Represents a 7-day revision plan."""
    plan: List[RevisionDay]
