import logging
import os
import time
from typing import Type, TypeVar, List, Tuple, Optional
from pydantic import BaseModel
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")
        
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call_structured(self, system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
        """Calls the LLM with structured outputs and exponential backoff retry for transient errors."""
        retries = 0
        last_error = None

        while retries <= self.max_retries:
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
                    # Refusal or unexpected output
                    refusal = getattr(response.choices[0].message, "refusal", None)
                    raise ValueError(f"LLM refused or returned empty output: {refusal}")
                return parsed

            except (RateLimitError, APITimeoutError) as e:
                retries += 1
                last_error = e
                if retries > self.max_retries:
                    break
                sleep_time = self.retry_delay * (2 ** (retries - 1))
                logger.warning(f"Transient error ({type(e).__name__}): {e}. Retrying in {sleep_time:.1f}s (attempt {retries}/{self.max_retries})...")
                time.sleep(sleep_time)

            except APIError as e:
                logger.error(f"OpenAI API Error: {e.message}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error during structured completion: {e}")
                raise

        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded for LLM completion. Last error: {last_error}")

    def get_summarization_prompt(self, context: str) -> Tuple[str, str]:
        system = (
            "You are a technical academic assistant. Summarize the provided engineering document chunks "
            "into topic-wise sections. Focus on key concepts, formulas, and definitions. "
            "Ground your summary strictly in the provided context."
        )
        user = f"Context:\n{context}\n\nGenerate a topic-wise summary."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> Tuple[str, str]:
        system = (
            "You are an examiner. Generate high-quality, exam-style MCQs based on the provided engineering content. "
            "Focus on conceptual understanding, not trivial details. "
            "Each MCQ must have 4 options, a correct answer, and a brief explanation."
        )
        user = f"Context:\n{context}\n\nGenerate {num_questions} MCQs."
        return system, user

    def get_planner_prompt(self, topics: List[str]) -> Tuple[str, str]:
        system = (
            "You are a study coach. Create a realistic 7-day revision plan for an engineering student based on the list of "
            "topics extracted from their notes. The plan should be time-boxed and include specific activities like "
            "'Review formulas', 'Solve practice problems', or 'Deep dive into theory'."
        )
        user = f"Topics:\n{', '.join(topics)}\n\nGenerate a 7-day revision plan."
        return system, user
