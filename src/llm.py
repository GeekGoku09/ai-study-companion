import os
from typing import List, Optional, Tuple, Type, TypeVar
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Centralized service for interacting with OpenAI structured outputs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService. Set OPENAI_API_KEY environment variable.")

        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def call_structured(self, system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
        """
        Calls the LLM with structured parsing and returns a Pydantic model instance.

        Raises:
            RuntimeError: If the OpenAI API call or response parsing fails.
        """
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("LLM response did not contain a valid parsed model.")
            return parsed
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate structured LLM response: {e}") from e

    def get_summarization_prompt(self, context: str) -> Tuple[str, str]:
        system = (
            "You are an expert technical academic assistant. Summarize the provided engineering document chunks "
            "into cohesive, topic-wise sections. Focus on key concepts, technical formulas, and precise definitions. "
            "Ground your summary strictly in the provided context and avoid speculative hallucinations."
        )
        user = f"Context:\n{context}\n\nGenerate a topic-wise summary."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> Tuple[str, str]:
        system = (
            "You are a rigorous engineering examiner. Generate high-quality, exam-style Multiple Choice Questions (MCQs) "
            "based strictly on the provided engineering content. Focus on conceptual understanding, numerical problems, and technical principles. "
            "Each MCQ must have exactly 4 choices (options), one unambiguously correct answer, and a clear explanation."
        )
        user = f"Context:\n{context}\n\nGenerate exactly {num_questions} MCQs based on this context."
        return system, user

    def get_planner_prompt(self, topics: List[str]) -> Tuple[str, str]:
        system = (
            "You are an engineering study coach. Create a realistic, time-boxed 7-day revision plan for an engineering "
            "student based on the provided list of topics extracted from their course material. "
            "Each day must specify the day number (1 to 7), the covered topics, actionable activities (e.g., 'Review formulas', "
            "'Derive theorems', 'Solve sample problems'), and an estimated study time in hours (realistic for a student)."
        )
        user = f"Topics:\n{', '.join(topics)}\n\nGenerate a 7-day revision plan."
        return system, user
