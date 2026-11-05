import os
import time
import logging
from typing import Type, TypeVar, List, Tuple, Optional
from pydantic import BaseModel
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for LLMService")
        
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T]
    ) -> T:
        """
        Calls OpenAI with structured parsing and exponential backoff retry logic
        for resilient handling of rate limits and transient connection issues.
        """
        last_error = None
        delay = self.backoff_factor

        for attempt in range(1, self.max_retries + 1):
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
                    refusal = getattr(response.choices[0].message, "refusal", None)
                    raise ValueError(f"LLM refused to generate structured response: {refusal}")
                return parsed

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_error = e
                logger.warning(
                    f"LLM API request failed on attempt {attempt}/{self.max_retries} with {type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    break
            except Exception as e:
                # Immediate re-raise for non-transient errors (e.g., authentication, bad schema)
                logger.error(f"Unrecoverable error in call_structured: {e}")
                raise

        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}") from last_error

    def get_summarization_prompt(self, context: str) -> Tuple[str, str]:
        system = (
            "You are a principal technical academic assistant and educator. "
            "Summarize the provided engineering document chunks into clear, topic-wise sections. "
            "Highlight core engineering concepts, definitions, formulas, and governing equations. "
            "Strictly ground your summary in the provided context and avoid outside hallucinations."
        )
        user = f"Context Material:\n{context}\n\nGenerate an in-depth topic-wise summary grounded in the context."
        return system, user

    def get_mcq_prompt(self, context: str, num_questions: int = 5) -> Tuple[str, str]:
        system = (
            "You are a rigorous engineering examiner. Generate challenging, exam-style Multiple Choice Questions (MCQs) "
            "based strictly on the provided context. "
            "Focus on conceptual understanding, problem solving, and technical accuracy. "
            "Each question MUST include exactly 4 distinct options, specify the correct answer matching one of the options, "
            "and provide a concise pedagogical explanation."
        )
        user = f"Context Material:\n{context}\n\nGenerate exactly {num_questions} high-quality MCQs."
        return system, user

    def get_planner_prompt(self, topics: List[str]) -> Tuple[str, str]:
        system = (
            "You are an expert engineering study coach. Create a realistic, high-impact 7-day revision plan "
            "covering the specified topics. "
            "Distribute topics logically across days (from fundamentals to advanced applications and mock review), "
            "include concrete actionable activities (such as 'Derive key equations', 'Solve practice problem sets', 'Review cheat sheet'), "
            "and assign realistic study hours (between 1.0 and 6.0 hours per day)."
        )
        topics_str = ", ".join(topics) if topics else "Engineering Core Foundations"
        user = f"Topics to cover:\n{topics_str}\n\nGenerate a structured 7-day revision plan."
        return system, user
