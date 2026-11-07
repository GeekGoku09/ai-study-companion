import json
import unittest

from src.schemas import (
    MCQ,
    MCQList,
    RevisionDay,
    RevisionPlan,
    SummaryResponse,
    TopicSummary,
)
from src.utils import (
    check_file_exists,
    count_tokens,
    format_mcq_markdown,
    format_plan_markdown,
    format_summary_markdown,
    format_tokens,
    serialize_model_json,
)


class TestUtils(unittest.TestCase):
    def test_token_counting(self):
        text = "Thermodynamics is the branch of physics that deals with heat and work."
        tokens = count_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(format_tokens(text), tokens)

    def test_check_file_exists(self):
        self.assertTrue(check_file_exists("README.md"))
        self.assertFalse(check_file_exists("non_existent_file_12345.xyz"))

    def test_format_summary_markdown(self):
        empty_summary = SummaryResponse(topics=[])
        self.assertIn("_No topics found._", format_summary_markdown(empty_summary))

        summary = SummaryResponse(topics=[
            TopicSummary(topic="Fluid Dynamics", summary="Study of fluids in motion.")
        ])
        md = format_summary_markdown(summary)
        self.assertIn("## Fluid Dynamics", md)
        self.assertIn("Study of fluids in motion.", md)

    def test_format_mcq_markdown(self):
        empty_mcqs = MCQList(questions=[])
        self.assertIn("_No questions generated._", format_mcq_markdown(empty_mcqs))

        mcqs = MCQList(questions=[
            MCQ(
                question="What is the unit of pressure?",
                options=["Pascal", "Joule", "Watt", "Newton"],
                correct_answer="Pascal",
                explanation="Pascal is N/m^2."
            )
        ])
        md = format_mcq_markdown(mcqs)
        self.assertIn("### Q1: What is the unit of pressure?", md)
        self.assertIn("- **A)** Pascal", md)
        self.assertIn("**Correct Answer:** Pascal", md)
        self.assertIn("**Explanation:** Pascal is N/m^2.", md)

    def test_format_plan_markdown(self):
        empty_plan = RevisionPlan(plan=[])
        self.assertIn("_No plan generated._", format_plan_markdown(empty_plan))

        plan = RevisionPlan(plan=[
            RevisionDay(
                day=1,
                topics=["Thermodynamics"],
                activities=["Review Carnot cycle", "Solve 5 practice problems"],
                estimated_time_hours=2.5,
            )
        ])
        md = format_plan_markdown(plan)
        self.assertIn("## Day 1", md)
        self.assertIn("Thermodynamics", md)
        self.assertIn("2.5 hours", md)
        self.assertIn("- Review Carnot cycle", md)

    def test_serialize_model_json(self):
        summary = SummaryResponse(topics=[
            TopicSummary(topic="Circuit Theory", summary="Ohm's law and Kirchhoff's laws.")
        ])
        json_str = serialize_model_json(summary)
        data = json.loads(json_str)
        self.assertEqual(data["topics"][0]["topic"], "Circuit Theory")


if __name__ == "__main__":
    unittest.main()
