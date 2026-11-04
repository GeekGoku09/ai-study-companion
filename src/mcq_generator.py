import logging
from typing import List
from .schemas import DocumentChunk, MCQList
from .llm import LLMService

logger = logging.getLogger(__name__)

class MCQGenerator:
    """Generates rigorous, grounded exam-style MCQs from retrieved chunks."""
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_mcqs(self, chunks: List[DocumentChunk], num_questions: int = 5) -> MCQList:
        """Generates MCQs from a list of document chunks."""
        if not chunks:
            logger.info("No chunks provided to generate_mcqs. Returning empty question list.")
            return MCQList(questions=[])

        if num_questions <= 0:
            return MCQList(questions=[])

        context_parts = []
        for i, c in enumerate(chunks, start=1):
            page_info = f"Page {c.metadata.get('page')}" if c.metadata.get("page") else f"Chunk {i}"
            context_parts.append(f"[{page_info}]\n{c.content.strip()}")

        context = "\n\n".join(context_parts)
        system_prompt, user_prompt = self.llm_service.get_mcq_prompt(context, num_questions=num_questions)
        
        mcq_list = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MCQList
        )
        
        return mcq_list
