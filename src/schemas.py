from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as source file, page, chunk_id")
    embedding: Optional[List[float]] = None

class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str = Field(..., min_length=5, description="The examination question")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Four possible answer choices")
    correct_answer: str = Field(..., description="The correct answer text or option label matching an option")
    explanation: str = Field(..., min_length=5, description="Detailed explanation of the correct answer")

    @model_validator(mode="after")
    def validate_correct_answer(self) -> "MCQ":
        # Check if correct_answer matches one of options directly,
        # or matches an option letter ('A', 'B', 'C', 'D') or prefix ('A)', 'B.', etc.)
        cleaned_ans = self.correct_answer.strip()
        option_texts = [opt.strip() for opt in self.options]

        if cleaned_ans in option_texts:
            return self

        # Allow single letter referencing options (A -> option 0, B -> option 1, etc.)
        letters = ["A", "B", "C", "D"]
        if cleaned_ans.upper() in letters:
            idx = letters.index(cleaned_ans.upper())
            self.correct_answer = option_texts[idx]
            return self

        # Check if correct_answer starts with letter prefix like "A) ..." or "A. ..."
        for idx, letter in enumerate(letters):
            if cleaned_ans.upper().startswith((f"{letter})", f"{letter}.", f"{letter}:")):
                self.correct_answer = option_texts[idx]
                return self

        # Case-insensitive substring match fallback
        for opt in option_texts:
            if cleaned_ans.lower() == opt.lower() or opt.lower().startswith(cleaned_ans.lower()):
                self.correct_answer = opt
                return self

        # If it doesn't match at all, keep as-is if non-empty, but raise ValueError if completely blank
        if not cleaned_ans:
            raise ValueError("correct_answer cannot be empty")
        return self

class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ] = Field(default_factory=list)

class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str = Field(..., min_length=1, description="Topic heading")
    summary: str = Field(..., min_length=10, description="Comprehensive summary of the topic")

class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary] = Field(default_factory=list)

class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int = Field(..., ge=1, le=30, description="Day number in study schedule")
    topics: List[str] = Field(default_factory=list, description="Topics scheduled for this day")
    activities: List[str] = Field(default_factory=list, description="Study activities to complete")
    estimated_time_hours: float = Field(..., ge=0.1, le=24.0, description="Estimated study time in hours")

class RevisionPlan(BaseModel):
    """Represents a structured revision plan."""
    plan: List[RevisionDay] = Field(default_factory=list)
