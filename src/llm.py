import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")
        
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def call_structured(self, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        """Calls the LLM and returns a structured response matching the Pydantic model."""
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
            raise ValueError("Failed to parse structured response from LLM.")
        return parsed

    def get_summarization_prompt(self, context: str) -> tuple[str, str]:
        system = (
            "You are a technical academic assistant. Summarize the provided engineering "
            "document chunks into topic-wise sections. Focus on key concepts, formulas, and definitions. "
            "Ground your summary strictly in the provided context."
        )
        user = f"Context:\n{context}\n\nGenerate a topic-wise summary."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> tuple[str, str]:
        system = (
            "You are an examiner. Generate high-quality, exam-style MCQs based on the provided "
            "engineering content. Focus on conceptual understanding, not trivial details. "
            "Each MCQ must have 4 options, a correct answer, and a brief explanation."
        )
        user = f"Context:\n{context}\n\nGenerate {num_questions} MCQs."
        return system, user

    def get_planner_prompt(self, topics: list[str]) -> tuple[str, str]:
        system = (
            "You are a study coach. Create a realistic 7-day revision plan for an engineering "
            "student based on the list of topics extracted from their notes. The plan should be "
            "time-boxed and include specific activities like 'Review formulas', 'Solve practice problems', "
            "or 'Deep dive into theory'."
        )
        user = f"Topics:\n{', '.join(topics)}\n\nGenerate a 7-day revision plan."
        return system, user
