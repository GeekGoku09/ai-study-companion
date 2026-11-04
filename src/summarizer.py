import logging
from typing import List
from .schemas import DocumentChunk, SummaryResponse
from .llm import LLMService

logger = logging.getLogger(__name__)

class Summarizer:
    """Generates comprehensive, grounded topic-wise summaries from retrieved chunks."""
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_summary(self, chunks: List[DocumentChunk]) -> SummaryResponse:
        """Generates a topic-wise summary from a list of document chunks."""
        if not chunks:
            logger.info("No chunks provided to generate_summary. Returning empty response.")
            return SummaryResponse(topics=[])

        context_parts = []
        for i, c in enumerate(chunks, start=1):
            page_info = f"Page {c.metadata.get('page')}" if c.metadata.get("page") else f"Chunk {i}"
            context_parts.append(f"[{page_info}]\n{c.content.strip()}")

        context = "\n\n".join(context_parts)
        system_prompt, user_prompt = self.llm_service.get_summarization_prompt(context)
        
        summary_response = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SummaryResponse
        )
        
        return summary_response
