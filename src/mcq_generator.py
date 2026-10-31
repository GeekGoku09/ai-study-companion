from typing import List
from .schemas import DocumentChunk, MCQList
from .llm import LLMService

class MCQGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_mcqs(self, chunks: List[DocumentChunk], num_questions: int = 5) -> MCQList:
        """Generates MCQs from a list of document chunks."""
        if not chunks or num_questions <= 0:
            return MCQList(questions=[])

        context_parts = []
        for c in chunks:
            page = c.metadata.get("page", "Unknown")
            context_parts.append(f"[Page {page}] {c.content.strip()}")
        context = "\n\n".join(context_parts)

        system_prompt, user_prompt = self.llm_service.get_mcq_prompt(context, num_questions=num_questions)
        
        mcq_list = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MCQList
        )
        
        return mcq_list
