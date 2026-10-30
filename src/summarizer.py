from typing import List, Optional
from .schemas import DocumentChunk, SummaryResponse
from .llm import LLMService
from .utils import setup_logger

logger = setup_logger(__name__)

class Summarizer:
    """Generates structured, topic-wise technical summaries grounded in retrieved context."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_summary(self, chunks: List[DocumentChunk]) -> SummaryResponse:
        """
        Generates a topic-wise summary from a list of document chunks.

        Args:
            chunks: List of DocumentChunk retrieved from context.

        Returns:
            SummaryResponse: Structured topic summaries.
        """
        if not chunks:
            logger.warning("No context chunks provided to Summarizer. Returning empty SummaryResponse.")
            return SummaryResponse(topics=[])

        context_parts = []
        for i, c in enumerate(chunks):
            page_info = f"Page {c.metadata.get('page')}" if c.metadata.get("page") else f"Section {i+1}"
            context_parts.append(f"[{page_info}]\n{c.content.strip()}")

        context = "\n\n".join(context_parts)
        system_prompt, user_prompt = self.llm_service.get_summarization_prompt(context)

        logger.info(f"Generating topic-wise summary using {len(chunks)} context chunk(s)...")
        summary_response = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SummaryResponse
        )

        return summary_response
