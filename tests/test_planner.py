import unittest
from unittest.mock import MagicMock
from src.planner import RevisionPlanner
from src.llm import LLMService
from src.schemas import RevisionPlan, RevisionDay

class TestPlanner(unittest.TestCase):
    def test_generate_plan_success(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_plan = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Calculus I"],
                    activities=["Review integration formulas", "Solve 5 practice problems"],
                    estimated_time_hours=2.5
                ),
                RevisionDay(
                    day=2,
                    topics=["Linear Algebra"],
                    activities=["Study matrix decomposition"],
                    estimated_time_hours=2.0
                )
            ]
        )
        mock_llm.get_planner_prompt.return_value = ("system", "user")
        mock_llm.call_structured.return_value = expected_plan

        planner = RevisionPlanner(mock_llm)
        topics = ["Calculus I", "Linear Algebra"]

        result = planner.generate_plan(topics)
        self.assertIsInstance(result, RevisionPlan)
        self.assertEqual(len(result.plan), 2)
        self.assertEqual(result.plan[0].day, 1)
        self.assertEqual(result.plan[0].estimated_time_hours, 2.5)

    def test_generate_plan_empty_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan([])
        self.assertEqual(len(result.plan), 0)

if __name__ == '__main__':
    unittest.main()
