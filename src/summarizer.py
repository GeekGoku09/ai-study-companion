import logging
from typing import List
from .schemas import DocumentChunk, SummaryResponse
from .llm import LLMService

logger = logging.getLogger(__name__)


class Summarizer:
    """Generates grounded topic summaries from retrieved context chunks."""
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_summary(self, chunks: List[DocumentChunk]) -> SummaryResponse:
        """Generates a topic-wise summary from a list of document chunks."""
        if not chunks:
            logger.warning("No chunks provided to Summarizer. Returning empty summary.")
            return SummaryResponse(topics=[])

        context = "\n\n".join([f"Source: Page {c.metadata.get('page', 'Unknown')}\nContent: {c.content}" for c in chunks])
        system_prompt, user_prompt = self.llm_service.get_summarization_prompt(context)

        summary_response = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SummaryResponse
        )

        return summary_response
