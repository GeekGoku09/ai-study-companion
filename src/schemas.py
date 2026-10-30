from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_answer: str
    explanation: str

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: str, info) -> str:
        options = info.data.get("options")
        if not options:
            return v

        v_stripped = v.strip()
        # Check if the answer matches one of the option texts exactly
        if v_stripped in options:
            return v_stripped

        # Check if the answer is specified as a letter (A, B, C, D)
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        upper_v = v_stripped.upper().rstrip(").")
        if upper_v in letter_map and letter_map[upper_v] < len(options):
            return options[letter_map[upper_v]]

        # Also accept prefix formats like "A) Option text"
        for opt in options:
            if v_stripped.lower() == opt.lower() or opt.lower().endswith(v_stripped.lower()):
                return opt

        # Fallback to returning original value to avoid breaking if LLM phrased slightly differently
        return v_stripped

class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ] = Field(default_factory=list)

class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str
    summary: str

class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary] = Field(default_factory=list)

class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int
    topics: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    estimated_time_hours: float = Field(default=2.0, ge=0.0)

class RevisionPlan(BaseModel):
    """Represents a 7-day revision plan."""
    plan: List[RevisionDay] = Field(default_factory=list)
