import json
import unittest
from src.utils import (
    check_file_exists,
    count_tokens,
    format_tokens,
    format_summary_markdown,
    format_mcq_markdown,
    format_plan_markdown,
    serialize_model_to_json,
)
from src.schemas import (
    SummaryResponse,
    TopicSummary,
    MCQList,
    MCQ,
    RevisionPlan,
    RevisionDay,
)


class TestUtils(unittest.TestCase):
    def test_check_file_exists(self):
        self.assertTrue(check_file_exists("README.md"))
        self.assertFalse(check_file_exists("non_existent_file.xyz"))
        self.assertFalse(check_file_exists(""))

    def test_count_tokens(self):
        self.assertEqual(count_tokens(""), 0)
        text = "Hello world! This is an AI study companion."
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertEqual(format_tokens(text), count)

    def test_format_summary_markdown(self):
        empty_summary = SummaryResponse(topics=[])
        self.assertIn("No topics extracted", format_summary_markdown(empty_summary))

        summary = SummaryResponse(
            topics=[TopicSummary(topic="Physics", summary="Study of matter and energy.")]
        )
        md = format_summary_markdown(summary)
        self.assertIn("## Physics", md)
        self.assertIn("Study of matter and energy.", md)

    def test_format_mcq_markdown(self):
        empty_mcqs = MCQList(questions=[])
        self.assertIn("No questions generated", format_mcq_markdown(empty_mcqs))

        mcqs = MCQList(
            questions=[
                MCQ(
                    question="What is the unit of force?",
                    options=["Joule", "Newton", "Watt", "Pascal"],
                    correct_answer="Newton",
                    explanation="Force is measured in Newtons.",
                )
            ]
        )
        md = format_mcq_markdown(mcq_list=mcqs)
        self.assertIn("### Q1: What is the unit of force?", md)
        self.assertIn("- **B)** Newton", md)
        self.assertIn("**Correct Answer:** Newton", md)

    def test_format_plan_markdown(self):
        empty_plan = RevisionPlan(plan=[])
        self.assertIn("No revision plan generated", format_plan_markdown(empty_plan))

        plan = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Fluid Dynamics"],
                    activities=["Study Bernoulli equation"],
                    estimated_time_hours=2.0,
                )
            ]
        )
        md = format_plan_markdown(plan)
        self.assertIn("## Day 1", md)
        self.assertIn("Fluid Dynamics", md)
        self.assertIn("2.0 hours", md)

    def test_serialize_model_to_json(self):
        summary = SummaryResponse(
            topics=[TopicSummary(topic="Test", summary="Desc")]
        )
        json_str = serialize_model_to_json(summary)
        data = json.loads(json_str)
        self.assertEqual(data["topics"][0]["topic"], "Test")


if __name__ == "__main__":
    unittest.main()
