from typing import List
from .schemas import DocumentChunk, SummaryResponse
from .llm import LLMService


class Summarizer:
    """Orchestrates topic-wise summarization from document chunks."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_summary(self, chunks: List[DocumentChunk]) -> SummaryResponse:
        """Generates a topic-wise summary from a list of document chunks."""
        if not chunks:
            return SummaryResponse(topics=[])

        context_parts = []
        for c in chunks:
            page = c.metadata.get("page", "Unknown")
            context_parts.append(f"[Page {page}]\n{c.content}")
        context = "\n\n".join(context_parts)

        system_prompt, user_prompt = self.llm_service.get_summarization_prompt(context)
        
        summary_response = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SummaryResponse
        )
        
        return summary_response
