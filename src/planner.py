import logging
from typing import List
from .schemas import RevisionPlan
from .llm import LLMService

logger = logging.getLogger(__name__)

class RevisionPlanner:
    def __init__(self, llm_service: LLMService):
        if not llm_service:
            raise ValueError("LLMService is required for RevisionPlanner")
        self.llm_service = llm_service

    def generate_plan(self, topics: List[str]) -> RevisionPlan:
        """Generates a 7-day revision plan based on a list of engineering topics."""
        cleaned_topics = [t.strip() for t in topics if t and t.strip()]
        if not cleaned_topics:
            logger.warning("No valid topics provided for revision plan.")
            return RevisionPlan(plan=[])

        system_prompt, user_prompt = self.llm_service.get_planner_prompt(cleaned_topics)
        
        revision_plan = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RevisionPlan
        )
        
        return revision_plan
