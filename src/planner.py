from typing import List
from .schemas import RevisionPlan
from .llm import LLMService
from .utils import setup_logger

logger = setup_logger(__name__)

class RevisionPlanner:
    """Generates structured 7-day study revision plans from extracted syllabus topics."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_plan(self, topics: List[str]) -> RevisionPlan:
        """
        Generates a 7-day revision plan based on a list of topics.

        Args:
            topics: List of topic names or themes.

        Returns:
            RevisionPlan: A 7-day scheduled breakdown.
        """
        cleaned_topics = [t.strip() for t in topics if t and t.strip()]
        if not cleaned_topics:
            logger.warning("No topics provided to RevisionPlanner. Returning empty RevisionPlan.")
            return RevisionPlan(plan=[])

        system_prompt, user_prompt = self.llm_service.get_planner_prompt(cleaned_topics)

        logger.info(f"Generating 7-day revision plan for {len(cleaned_topics)} topic(s)...")
        revision_plan = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RevisionPlan
        )

        return revision_plan
