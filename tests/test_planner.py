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
        self.assertIsInstance(result, RevisionPlan)

    def test_generate_plan_valid(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_plan = RevisionPlan(plan=[
            RevisionDay(
                day=1,
                topics=["Thermodynamics"],
                activities=["Review first law", "Solve 5 practice problems"],
                estimated_time_hours=2.5
            )
        ])
        mock_llm.get_planner_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected_plan

        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan(["Thermodynamics"])

        self.assertEqual(len(result.plan), 1)
        self.assertEqual(result.plan[0].day, 1)
        self.assertEqual(result.plan[0].topics[0], "Thermodynamics")
        md = result.to_markdown()
        self.assertIn("## Day 1", md)
        self.assertIn("Thermodynamics", md)

if __name__ == "__main__":
    unittest.main()
