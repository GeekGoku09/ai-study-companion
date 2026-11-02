import unittest
from unittest.mock import MagicMock
from src.planner import RevisionPlanner
from src.llm import LLMService
from src.schemas import RevisionPlan, RevisionDay

class TestPlanner(unittest.TestCase):
    def test_generate_plan_empty_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan([])
        self.assertEqual(len(result.plan), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_plan_success(self):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.get_planner_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Thermodynamics"],
                    activities=["Read chapter 1", "Solve 5 problems"],
                    estimated_time_hours=3.0
                )
            ]
        )
        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan(["Thermodynamics"])
        self.assertEqual(len(result.plan), 1)
        self.assertEqual(result.plan[0].day, 1)
        self.assertEqual(result.plan[0].estimated_time_hours, 3.0)
        mock_llm.call_structured.assert_called_once()

if __name__ == "__main__":
    unittest.main()
