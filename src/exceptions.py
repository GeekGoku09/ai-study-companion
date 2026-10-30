"""Custom domain exceptions for the RAG system."""

class StudyCompanionError(Exception):
    """Base exception for all domain-specific errors."""
    pass

class DocumentProcessingError(StudyCompanionError):
    """Raised when an error occurs during document ingestion or parsing."""
    pass

class VectorIndexError(StudyCompanionError):
    """Raised when an error occurs in building, querying, or saving/loading vector indexes."""
    pass

class LLMServiceError(StudyCompanionError):
    """Raised when communication with the LLM or structured parsing fails."""
    pass
