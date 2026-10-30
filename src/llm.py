import os
from typing import Type, TypeVar, List, Tuple, Optional
from pydantic import BaseModel
from openai import OpenAI
from .exceptions import LLMServiceError
from .utils import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMService:
    """Centralized LLM client interface supporting structured output and customizable models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")

        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def call_structured(self, system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
        """
        Calls the LLM using OpenAI structured outputs with response_model parsing.

        Raises:
            LLMServiceError: If the API request fails or parsed output is invalid.
        """
        if not system_prompt or not user_prompt:
            raise LLMServiceError("system_prompt and user_prompt must not be empty.")

        try:
            logger.debug(f"Calling structured LLM completion with model {self.model}")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                # If parsed is None, the model may have refused the request
                refusal = getattr(response.choices[0].message, "refusal", None)
                raise LLMServiceError(f"Model refused request or returned empty parsed model: {refusal}")
            return parsed
        except Exception as e:
            logger.error(f"Structured LLM call failed: {e}")
            raise LLMServiceError(f"Failed to generate structured response: {e}") from e

    def get_summarization_prompt(self, context: str) -> Tuple[str, str]:
        system = (
            "You are an expert technical academic assistant and tutor. "
            "Summarize the provided engineering document chunks into clear, topic-wise sections. "
            "Focus on core concepts, key formulas, architectural principles, and exact definitions. "
            "You MUST ground your summary strictly in the provided context and avoid outside assumptions."
        )
        user = f"Context:\n{context}\n\nGenerate a detailed topic-wise summary grounded strictly in this context."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> Tuple[str, str]:
        system = (
            "You are a rigorous engineering examiner. Generate high-quality, exam-style Multiple Choice Questions (MCQs) "
            "based strictly on the provided technical content. "
            "Focus on conceptual understanding, problem-solving, and formula application. "
            "Each question MUST provide exactly 4 distinct options, designate the correct answer, and include a clear explanation."
        )
        user = f"Context:\n{context}\n\nGenerate {num_questions} exam-style MCQs."
        return system, user

    def get_planner_prompt(self, topics: List[str]) -> Tuple[str, str]:
        system = (
            "You are a disciplined engineering study coach. "
            "Create a realistic, time-boxed 7-day revision plan based on the provided list of topics. "
            "Each day must specify the target topics, actionable study activities (e.g., 'Review formulas', "
            "'Solve practice problems', 'Deep-dive theory'), and estimated study hours."
        )
        user = f"Topics:\n{', '.join(topics)}\n\nGenerate a 7-day revision plan covering these topics effectively."
        return system, user
