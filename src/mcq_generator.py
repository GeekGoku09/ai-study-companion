from typing import List
from .schemas import DocumentChunk, MCQList
from .llm import LLMService
from .utils import setup_logger

logger = setup_logger(__name__)

class MCQGenerator:
    """Generates exam-style Multiple Choice Questions grounded in retrieved document chunks."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_mcqs(self, chunks: List[DocumentChunk], num_questions: int = 5) -> MCQList:
        """
        Generates MCQs from a list of document chunks.

        Args:
            chunks: Document chunks serving as grounding context.
            num_questions: Target number of MCQs to generate (default: 5).

        Returns:
            MCQList: Pydantic model containing generated questions.
        """
        if not chunks:
            logger.warning("No context chunks provided to MCQGenerator. Returning empty MCQList.")
            return MCQList(questions=[])

        num_questions = max(1, num_questions)
        context = "\n\n".join([f"Content:\n{c.content.strip()}" for c in chunks])
        system_prompt, user_prompt = self.llm_service.get_mcq_prompt(context, num_questions=num_questions)

        logger.info(f"Generating {num_questions} MCQ(s) from {len(chunks)} context chunk(s)...")
        mcq_list = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MCQList
        )

        return mcq_list
