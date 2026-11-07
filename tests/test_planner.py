import unittest
from unittest.mock import MagicMock

from src.llm import LLMService
from src.planner import RevisionPlanner
from src.schemas import RevisionDay, RevisionPlan


class TestRevisionPlanner(unittest.TestCase):
    def test_generate_plan_success(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_plan = RevisionPlan(plan=[
            RevisionDay(
                day=1,
                topics=["Linear Algebra"],
                activities=["Review matrix inversion"],
                estimated_time_hours=2.0,
            )
        ])
        mock_llm.get_planner_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected_plan

        planner = RevisionPlanner(mock_llm)
        topics = ["Linear Algebra", "Calculus"]
        result = planner.generate_plan(topics)

        self.assertEqual(len(result.plan), 1)
        self.assertEqual(result.plan[0].day, 1)
        mock_llm.call_structured.assert_called_once()

    def test_generate_plan_empty_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan([])
        self.assertEqual(result.plan, [])
        mock_llm.call_structured.assert_not_called()


if __name__ == "__main__":
    unittest.main()
