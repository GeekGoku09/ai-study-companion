from .llm import LLMService
from .schemas import DocumentChunk, MCQList


class MCQGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_mcqs(self, chunks: list[DocumentChunk], num_questions: int = 5) -> MCQList:
        """Generates MCQs from a list of document chunks."""
        if not chunks:
            return MCQList(questions=[])

        context = "\n\n".join([f"Content: {c.content}" for c in chunks])
        system_prompt, user_prompt = self.llm_service.get_mcq_prompt(context, num_questions)
        
        mcq_list = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MCQList,
        )
        
        return mcq_list
