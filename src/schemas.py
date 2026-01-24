from pydantic import BaseModel, Field
from typing import List, Optional

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
    day: int
    topics: List[str]
    activities: List[str]
    estimated_time_hours: float

class RevisionPlan(BaseModel):
    """Represents a 7-day revision plan."""
    plan: List[RevisionDay]
