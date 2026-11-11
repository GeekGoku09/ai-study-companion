import unittest
from unittest.mock import MagicMock
from src.llm import LLMService
from src.planner import RevisionPlanner
from src.schemas import RevisionPlan, RevisionDay


class TestRevisionPlanner(unittest.TestCase):
    def test_generate_plan_empty_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan([])
        self.assertEqual(len(result.plan), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_plan_with_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_plan = RevisionPlan(plan=[
            RevisionDay(day=1, topics=["Calculus"], activities=["Review derivatives"], estimated_time_hours=2.0),
            RevisionDay(day=2, topics=["Linear Algebra"], activities=["Solve matrix problems"], estimated_time_hours=3.0),
        ])
        mock_llm.get_planner_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected_plan

        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan(["Calculus", "Linear Algebra"])
        self.assertEqual(len(result.plan), 2)
        self.assertEqual(result.plan[0].day, 1)
        self.assertEqual(result.plan[0].estimated_time_hours, 2.0)
        mock_llm.call_structured.assert_called_once()


if __name__ == "__main__":
    unittest.main()
