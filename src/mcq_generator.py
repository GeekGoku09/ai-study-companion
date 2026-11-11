import logging
from typing import List
from .schemas import DocumentChunk, MCQList
from .llm import LLMService

logger = logging.getLogger(__name__)


class MCQGenerator:
    """Generates exam-style multiple choice questions grounded in document context."""
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_mcqs(self, chunks: List[DocumentChunk], num_questions: int = 5) -> MCQList:
        """Generates MCQs from a list of document chunks."""
        if not chunks:
            logger.warning("No chunks provided to MCQGenerator. Returning empty question list.")
            return MCQList(questions=[])

        context = "\n\n".join([f"Source: Page {c.metadata.get('page', 'Unknown')}\nContent: {c.content}" for c in chunks])
        system_prompt, user_prompt = self.llm_service.get_mcq_prompt(context, num_questions=max(1, num_questions))

        mcq_list = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MCQList
        )

        return mcq_list
