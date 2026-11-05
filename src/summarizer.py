import logging
from typing import List
from .schemas import DocumentChunk, SummaryResponse
from .llm import LLMService

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, llm_service: LLMService):
        if not llm_service:
            raise ValueError("LLMService is required for Summarizer")
        self.llm_service = llm_service

    def generate_summary(self, chunks: List[DocumentChunk]) -> SummaryResponse:
        """Generates a topic-wise summary from a list of document chunks."""
        if not chunks:
            logger.warning("No chunks provided for summarization.")
            return SummaryResponse(topics=[])

        context_parts = []
        for i, c in enumerate(chunks, start=1):
            page_info = c.metadata.get("page", "Unknown")
            context_parts.append(f"[Section {i} - Page {page_info}]\n{c.content.strip()}")

        context = "\n\n".join(context_parts)
        system_prompt, user_prompt = self.llm_service.get_summarization_prompt(context)
        
        summary_response = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SummaryResponse
        )
        
        return summary_response
