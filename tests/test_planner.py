import unittest
from unittest.mock import MagicMock

from src.llm import LLMService
from src.planner import RevisionPlanner
from src.schemas import RevisionDay, RevisionPlan


class TestRevisionPlanner(unittest.TestCase):
    def test_generate_plan_empty_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        planner = RevisionPlanner(mock_llm)

        result = planner.generate_plan([])
        self.assertIsInstance(result, RevisionPlan)
        self.assertEqual(len(result.plan), 0)

        result_whitespace = planner.generate_plan(["   ", ""])
        self.assertEqual(len(result_whitespace.plan), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_plan_success(self):
        mock_llm = MagicMock(spec=LLMService)
        expected = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Thermodynamics"],
                    activities=["Review formulas", "Solve sample problems"],
                    estimated_time_hours=2.5,
                )
            ]
        )
        mock_llm.get_planner_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected

        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan(["Thermodynamics"])

        self.assertEqual(len(result.plan), 1)
        self.assertEqual(result.plan[0].day, 1)
        self.assertEqual(result.plan[0].estimated_time_hours, 2.5)


if __name__ == "__main__":
    unittest.main()
