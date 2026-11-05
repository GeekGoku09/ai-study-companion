import unittest
from unittest.mock import MagicMock
from src.planner import RevisionPlanner
from src.llm import LLMService
from src.schemas import RevisionPlan, RevisionDay

class TestPlanner(unittest.TestCase):
    def test_generate_plan_empty_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        planner = RevisionPlanner(mock_llm)
        res = planner.generate_plan([])
        self.assertIsInstance(res, RevisionPlan)
        self.assertEqual(len(res.plan), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_plan_valid_topics(self):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.get_planner_prompt.return_value = ("sys", "user")
        expected = RevisionPlan(plan=[
            RevisionDay(day=1, topics=["Signals"], activities=["Review Fourier Transform"], estimated_time_hours=2.0)
        ])
        mock_llm.call_structured.return_value = expected

        planner = RevisionPlanner(mock_llm)
        result = planner.generate_plan(["Signals"])

        self.assertEqual(len(result.plan), 1)
        self.assertEqual(result.plan[0].day, 1)
        mock_llm.call_structured.assert_called_once()

    def test_planner_none_service_raises(self):
        with self.assertRaises(ValueError):
            RevisionPlanner(None)

if __name__ == '__main__':
    unittest.main()
