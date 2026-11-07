from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = None


class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str
    options: list[str] = Field(..., min_length=4, max_length=4)
    correct_answer: str
    explanation: str


class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: list[MCQ]


class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str
    summary: str


class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: list[TopicSummary]


class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int
    topics: list[str]
    activities: list[str]
    estimated_time_hours: float


class RevisionPlan(BaseModel):
    """Represents a 7-day revision plan."""
    plan: list[RevisionDay]
