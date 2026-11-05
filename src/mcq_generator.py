import logging
from typing import List
from .schemas import DocumentChunk, MCQList
from .llm import LLMService

logger = logging.getLogger(__name__)

class MCQGenerator:
    def __init__(self, llm_service: LLMService):
        if not llm_service:
            raise ValueError("LLMService is required for MCQGenerator")
        self.llm_service = llm_service

    def generate_mcqs(self, chunks: List[DocumentChunk], num_questions: int = 5) -> MCQList:
        """Generates exam-style MCQs strictly grounded in the document chunks."""
        if not chunks:
            logger.warning("No chunks provided for MCQ generation.")
            return MCQList(questions=[])

        num_questions = max(1, min(50, num_questions))
        context_parts = []
        for i, c in enumerate(chunks, start=1):
            page_info = c.metadata.get("page", "Unknown")
            context_parts.append(f"[Excerpt {i} - Page {page_info}]\n{c.content.strip()}")

        context = "\n\n".join(context_parts)
        system_prompt, user_prompt = self.llm_service.get_mcq_prompt(context, num_questions=num_questions)
        
        mcq_list = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MCQList
        )
        
        return mcq_list
