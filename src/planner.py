from typing import List
from .schemas import RevisionPlan
from .llm import LLMService

class RevisionPlanner:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_plan(self, topics: List[str]) -> RevisionPlan:
        """Generates a 7-day revision plan based on a list of topics."""
        if not topics:
            return RevisionPlan(plan=[])

        system_prompt, user_prompt = self.llm_service.get_planner_prompt(topics)
        
        revision_plan = self.llm_service.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RevisionPlan
        )
        
        return revision_plan
