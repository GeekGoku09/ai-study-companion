from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty or blank")
        return v

class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_answer: str
    explanation: str

    @field_validator("options")
    @classmethod
    def validate_options(cls, options: List[str]) -> List[str]:
        cleaned = [opt.strip() for opt in options]
        if any(len(opt) == 0 for opt in cleaned):
            raise ValueError("MCQ options cannot be empty or whitespace-only")
        if len(set(cleaned)) < 4:
            raise ValueError("MCQ options must contain 4 distinct choices")
        return cleaned

    @model_validator(mode="after")
    def validate_correct_answer(self) -> "MCQ":
        # Check if correct_answer matches one of the options (either exact match or letter A/B/C/D)
        valid_options = self.options
        matched = False
        if self.correct_answer in valid_options:
            matched = True
        elif len(self.correct_answer.strip()) == 1 and self.correct_answer.upper() in ["A", "B", "C", "D"]:
            idx = ord(self.correct_answer.upper()) - ord("A")
            # Convert letter designation to matching option string for unambiguous downstream use
            self.correct_answer = self.options[idx]
            matched = True
        else:
            # Check case-insensitive match
            for opt in valid_options:
                if self.correct_answer.strip().lower() == opt.lower():
                    self.correct_answer = opt
                    matched = True
                    break

        if not matched:
            raise ValueError(
                f"correct_answer '{self.correct_answer}' does not match any of the 4 options: {valid_options}"
            )
        return self

class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ]

class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str
    summary: str

    @field_validator("topic", "summary")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or blank")
        return v.strip()

class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary]

class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int = Field(..., ge=1)
    topics: List[str]
    activities: List[str]
    estimated_time_hours: float = Field(..., gt=0.0)

class RevisionPlan(BaseModel):
    """Represents a structured revision plan."""
    plan: List[RevisionDay]
