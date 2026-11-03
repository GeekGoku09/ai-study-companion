import os
import time
import logging
from typing import Type, TypeVar, List, Tuple, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMServiceError(Exception):
    """Base exception for errors during LLM processing."""
    pass

class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")
        
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T]
    ) -> T:
        """
        Calls the LLM with structured outputs using exponential backoff retry.
        Catches API and parsing errors and wraps them in LLMServiceError.
        """
        if not user_prompt or not user_prompt.strip():
            raise ValueError("User prompt cannot be empty")

        attempts = 0
        last_exception = None

        while attempts < self.max_retries:
            attempts += 1
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
                    # Model refused or returned non-parseable response
                    refusal = getattr(response.choices[0].message, "refusal", None)
                    raise LLMServiceError(f"Model refused or failed structured response: {refusal}")
                return parsed

            except (RateLimitError, APITimeoutError) as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempts
                logger.warning(
                    f"LLM request encounter transient error ({e.__class__.__name__}). "
                    f"Retrying attempt {attempts}/{self.max_retries} after {wait_time:.1f}s."
                )
                time.sleep(wait_time)
            except ValidationError as e:
                raise LLMServiceError(f"Output failed schema validation: {e}") from e
            except APIError as e:
                raise LLMServiceError(f"OpenAI API error occurred: {e}") from e
            except Exception as e:
                raise LLMServiceError(f"Unexpected error during structured generation: {e}") from e

        raise LLMServiceError(
            f"Failed to obtain structured response after {self.max_retries} attempts: {last_exception}"
        ) from last_exception

    def get_summarization_prompt(self, context: str) -> Tuple[str, str]:
        system = (
            "You are a technical academic assistant. Summarize the provided engineering document chunks into "
            "topic-wise sections. Focus on key concepts, formulas, and definitions. Ground your summary strictly in "
            "the provided context. Do not invent details not present in the text."
        )
        user = f"Context:\n{context}\n\nGenerate a topic-wise summary."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> Tuple[str, str]:
        system = (
            "You are an engineering examiner. Generate high-quality, exam-style MCQs based strictly on the provided content. "
            "Focus on conceptual understanding, formulas, and definitions. Each MCQ must have exactly 4 distinct options, "
            "a correct answer matching one of the options, and a concise explanation."
        )
        user = f"Context:\n{context}\n\nGenerate {num_questions} MCQs."
        return system, user

    def get_planner_prompt(self, topics: List[str]) -> Tuple[str, str]:
        topics_str = ", ".join(topics) if topics else "Engineering Fundamentals"
        system = (
            "You are a study coach for engineering students. Create a realistic, time-boxed 7-day revision plan "
            "based on the list of topics provided. Include specific activities like 'Review formulas', "
            "'Solve practice problems', or 'Deep dive into theory'."
        )
        user = f"Topics:\n{topics_str}\n\nGenerate a 7-day revision plan."
        return system, user
