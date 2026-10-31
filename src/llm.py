import os
import time
from typing import Type, TypeVar, List, Tuple, Optional
from pydantic import BaseModel
from openai import OpenAI
from .schemas import DocumentChunk
from .utils import count_tokens, get_tokenizer

T = TypeVar("T", bound=BaseModel)

class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_context_tokens: int = 12000,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")
        
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.max_context_tokens = max_context_tokens
        self.max_retries = max_retries
        self.client = OpenAI(api_key=self.api_key)

    def truncate_context_to_budget(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncates context text if it exceeds maximum allowable prompt tokens."""
        limit = max_tokens or self.max_context_tokens
        if count_tokens(text) <= limit:
            return text

        encoding = get_tokenizer("cl100k_base")
        encoded = encoding.encode(text)
        truncated_encoded = encoded[:limit]
        return encoding.decode(truncated_encoded) + "\n\n[Context truncated to adhere to token budget]"

    def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T]
    ) -> T:
        """Calls the LLM with automatic retries for transient errors and returns structured response."""
        last_error = None
        for attempt in range(self.max_retries):
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
                    raise ValueError("LLM responded with empty parsed content.")
                return parsed
            except Exception as e:
                last_error = e
                # Do not retry on validation failures of the output schema if raised internally
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM structured call failed after {self.max_retries} attempts: {e}") from e

        raise RuntimeError(f"LLM structured call failed: {last_error}")

    def get_summarization_prompt(self, context: str) -> Tuple[str, str]:
        system = (
            "You are a technical academic assistant. Summarize the provided engineering document chunks into "
            "topic-wise sections. Focus on key concepts, formulas, and definitions. Ground your summary strictly "
            "in the provided context."
        )
        safe_context = self.truncate_context_to_budget(context)
        user = f"Context:\n{safe_context}\n\nGenerate a topic-wise summary."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> Tuple[str, str]:
        system = (
            "You are an examiner. Generate high-quality, exam-style MCQs based on the provided engineering content. "
            "Focus on conceptual understanding, not trivial details. Each MCQ must have exactly 4 options, a correct answer "
            "matching one of the options, and a brief explanation."
        )
        safe_context = self.truncate_context_to_budget(context)
        user = f"Context:\n{safe_context}\n\nGenerate {num_questions} MCQs."
        return system, user

    def get_planner_prompt(self, topics: List[str]) -> Tuple[str, str]:
        system = (
            "You are a study coach. Create a realistic 7-day revision plan for an engineering student based on "
            "the list of topics extracted from their notes. The plan should be time-boxed and include specific activities "
            "like 'Review formulas', 'Solve practice problems', or 'Deep dive into theory'."
        )
        cleaned_topics = [t.strip() for t in topics if t.strip()]
        user = f"Topics:\n{', '.join(cleaned_topics)}\n\nGenerate a 7-day revision plan."
        return system, user
