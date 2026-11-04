from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as source file, page number, chunk id")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content cannot be empty or whitespace only")
        return v

class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str = Field(..., min_length=5, description="Question text")
    options: List[str] = Field(..., min_length=4, max_length=4, description="List of exactly 4 choices")
    correct_answer: str = Field(..., min_length=1, description="The correct answer text or option")
    explanation: str = Field(..., min_length=5, description="Explanation for the correct answer")

    @model_validator(mode="after")
    def validate_correct_answer(self) -> "MCQ":
        # Normalize comparison: allow exact match or letter indicator like "A", "B", "C", "D"
        trimmed_ans = self.correct_answer.strip()
        trimmed_options = [opt.strip() for opt in self.options]

        if trimmed_ans in trimmed_options:
            return self

        # If answer is given as option letter A, B, C, D (or A), B), etc.)
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        clean_key = trimmed_ans.upper().rstrip(".)")
        if clean_key in letter_map:
            # Map letter to actual option content
            idx = letter_map[clean_key]
            self.correct_answer = self.options[idx]
            return self

        # Also check if option starts with the answer (e.g. "A) text" or "text")
        for opt in self.options:
            if opt.strip().lower() == trimmed_ans.lower():
                self.correct_answer = opt
                return self

        raise ValueError(
            f"correct_answer '{self.correct_answer}' does not match any of the provided options: {self.options}"
        )

class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ] = Field(default_factory=list, description="List of generated MCQs")

class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str = Field(..., min_length=1, description="Topic or section title")
    summary: str = Field(..., min_length=5, description="Summary details for the topic")

class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary] = Field(default_factory=list, description="List of topic summaries")

class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int = Field(..., ge=1, le=30, description="Day number in revision schedule")
    topics: List[str] = Field(..., min_length=1, description="List of topics covered on this day")
    activities: List[str] = Field(..., min_length=1, description="Specific revision tasks")
    estimated_time_hours: float = Field(..., gt=0.0, le=24.0, description="Estimated study time in hours")

class RevisionPlan(BaseModel):
    """Represents a revision plan."""
    plan: List[RevisionDay] = Field(default_factory=list, description="Daily revision schedule")
