import logging
import os
import time
from typing import TypeVar

from openai import APIConnectionError, OpenAI, RateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float | None = None,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")

        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries

    def call_structured(self, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        """Calls the LLM and returns a structured response matching the Pydantic model with retry logic."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
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
                    # In rare cases with content moderation or refusal
                    refusal = getattr(response.choices[0].message, "refusal", None)
                    raise ValueError(f"Model refused or failed to parse response: {refusal}")
                return parsed
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = 2**attempt
                    logger.warning(
                        "OpenAI API error (%s). Retrying in %ds (attempt %d/%d)...",
                        e,
                        backoff,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(backoff)
                else:
                    raise
            except Exception as e:
                raise ValueError(f"LLM structured completion failed: {e}") from e

        if last_error:
            raise last_error
        raise RuntimeError("Unexpected failure in LLM completion")

    def get_summarization_prompt(self, context: str) -> tuple[str, str]:
        system = (
            "You are a technical academic assistant. Summarize the provided engineering document chunks "
            "into topic-wise sections. Focus on key concepts, formulas, and definitions. Ground your summary "
            "strictly in the provided context."
        )
        user = f"Context:\n{context}\n\nGenerate a topic-wise summary."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> tuple[str, str]:
        system = (
            "You are an examiner. Generate high-quality, exam-style MCQs based on the provided engineering "
            "content. Focus on conceptual understanding, not trivial details. Each MCQ must have 4 options, "
            "a correct answer, and a brief explanation."
        )
        user = f"Context:\n{context}\n\nGenerate {num_questions} MCQs."
        return system, user

    def get_planner_prompt(self, topics: list[str]) -> tuple[str, str]:
        system = (
            "You are a study coach. Create a realistic 7-day revision plan for an engineering student based "
            "on the list of topics extracted from their notes. The plan should be time-boxed and include "
            "specific activities like 'Review formulas', 'Solve practice problems', or 'Deep dive into theory'."
        )
        user = f"Topics:\n{', '.join(topics)}\n\nGenerate a 7-day revision plan."
        return system, user
