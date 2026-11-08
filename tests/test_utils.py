import json
import os
import tempfile
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
    format_output_as_json,
    format_output_as_markdown,
    format_tokens,
    validate_pdf_path,
)


class TestUtils(unittest.TestCase):
    def test_count_tokens(self):
        self.assertEqual(count_tokens(""), 0)
        tokens = count_tokens("Hello, world! This is a test sentence.")
        self.assertGreater(tokens, 0)
        self.assertEqual(format_tokens("Hello, world!"), count_tokens("Hello, world!"))

    def test_file_exists_and_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            temp_pdf = f.name

        try:
            self.assertTrue(check_file_exists(temp_pdf))
            validate_pdf_path(temp_pdf)  # should not raise

            with self.assertRaises(ValueError):
                validate_pdf_path("not_a_pdf.txt")

            with self.assertRaises(ValueError):
                validate_pdf_path("")

            with self.assertRaises(FileNotFoundError):
                validate_pdf_path("non_existent_file.pdf")
        finally:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    def test_format_output_as_json(self):
        summary = SummaryResponse(
            topics=[TopicSummary(topic="Calculus", summary="Derivatives and Integrals")]
        )
        json_str = format_output_as_json(summary)
        data = json.loads(json_str)
        self.assertEqual(data["topics"][0]["topic"], "Calculus")

    def test_format_output_as_markdown(self):
        summary = SummaryResponse(
            topics=[TopicSummary(topic="Calculus", summary="Derivatives and Integrals")]
        )
        md_summary = format_output_as_markdown("summary", summary)
        self.assertIn("# Topic-Wise Summary", md_summary)
        self.assertIn("## Calculus", md_summary)

        mcqs = MCQList(
            questions=[
                MCQ(
                    question="What is derivative of x^2?",
                    options=["x", "2x", "x^2", "2"],
                    correct_answer="2x",
                    explanation="Power rule.",
                )
            ]
        )
        md_mcqs = format_output_as_markdown("mcq", mcqs)
        self.assertIn("# Exam Practice MCQs", md_mcqs)
        self.assertIn("- **B)** 2x", md_mcqs)

        plan = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Limits"],
                    activities=["Solve 10 problems"],
                    estimated_time_hours=1.5,
                )
            ]
        )
        md_plan = format_output_as_markdown("plan", plan)
        self.assertIn("# 7-Day Revision Plan", md_plan)
        self.assertIn("## Day 1 (1.5 hours)", md_plan)


if __name__ == "__main__":
    unittest.main()
