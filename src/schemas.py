from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: dict = Field(default_factory=dict, description="Metadata such as source, page, chunk_id")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the chunk content")


class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str = Field(..., min_length=1, description="Question prompt")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 choices")
    correct_answer: str = Field(..., min_length=1, description="The correct answer text or option label")
    explanation: str = Field(..., min_length=1, description="Explanation for the correct answer")


class MCQList(BaseModel):
    """Represents a list of Multiple Choice Questions."""
    questions: List[MCQ] = Field(default_factory=list, description="List of generated MCQs")


class TopicSummary(BaseModel):
    """Represents a summary for a specific topic."""
    topic: str = Field(..., min_length=1, description="Name or title of the topic")
    summary: str = Field(..., min_length=1, description="Detailed summary of the topic")


class SummaryResponse(BaseModel):
    """Represents the complete summarization output."""
    topics: List[TopicSummary] = Field(default_factory=list, description="List of topic summaries")


class RevisionDay(BaseModel):
    """Represents a single day in the revision plan."""
    day: int = Field(..., ge=1, description="Day number of the study plan")
    topics: List[str] = Field(..., description="Topics scheduled for this day")
    activities: List[str] = Field(..., description="Actionable study tasks/activities")
    estimated_time_hours: float = Field(..., ge=0, description="Estimated study duration in hours")


class RevisionPlan(BaseModel):
    """Represents a 7-day revision plan."""
    plan: List[RevisionDay] = Field(default_factory=list, description="Schedule of revision days")
