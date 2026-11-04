import logging
from typing import List
from .schemas import RevisionPlan
from .llm import LLMService

logger = logging.getLogger(__name__)

class RevisionPlanner:
    """Generates structured, time-boxed 7-day revision plans from topic lists."""
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_plan(self, topics: List[str]) -> RevisionPlan:
        """Generates a 7-day revision plan based on a list of topics."""
        clean_topics = [t.strip() for t in topics if t and t.strip()]
        if not clean_topics:
            logger.info("No valid topics provided to generate_plan. Returning empty plan.")
            return RevisionPlan(plan=[])

        system_prompt, user_prompt = self.llm_service.get_planner_prompt(clean_topics)
        
        revision_plan = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RevisionPlan
        )
        
        return revision_plan
