import os
import tempfile
import unittest
from src.schemas import MCQ, MCQList, RevisionDay, RevisionPlan, SummaryResponse, TopicSummary
from src.utils import count_tokens, export_to_json, export_to_markdown, format_tokens, write_output_file


class TestUtils(unittest.TestCase):
    def test_token_counting(self):
        text = "Hello world! This is a test sentence for token counting."
        count = count_tokens(text)
        self.assertGreater(count, 0)
        # Test backward-compatible alias
        self.assertEqual(format_tokens(text), count)
        # Empty text
        self.assertEqual(count_tokens(""), 0)

    def test_export_to_json_and_markdown_summary(self):
        summary = SummaryResponse(topics=[
            TopicSummary(topic="Physics", summary="Study of matter and energy.")
        ])
        json_str = export_to_json(summary)
        self.assertIn('"Physics"', json_str)

        md_str = export_to_markdown(summary)
        self.assertIn("# Topic-Wise Study Summary", md_str)
        self.assertIn("## Physics", md_str)

    def test_export_to_markdown_mcq(self):
        mcqs = MCQList(questions=[
            MCQ(
                question="What is the unit of Force?",
                options=["Newton", "Joule", "Pascal", "Watt"],
                correct_answer="Newton",
                explanation="Force is measured in Newtons (N)."
            )
        ])
        md_str = export_to_markdown(mcqs)
        self.assertIn("# Exam Practice Questions (MCQs)", md_str)
        self.assertIn("What is the unit of Force?", md_str)
        self.assertIn("**A)** Newton", md_str)

    def test_export_to_markdown_plan(self):
        plan = RevisionPlan(plan=[
            RevisionDay(day=1, topics=["Signals"], activities=["Review transforms"], estimated_time_hours=2.5)
        ])
        md_str = export_to_markdown(plan)
        self.assertIn("# 7-Day Revision Plan", md_str)
        self.assertIn("## Day 1 (2.5 hrs)", md_str)

    def test_write_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "subdir", "output.md")
            write_output_file("# Test Output Content", file_path)
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "# Test Output Content")


if __name__ == "__main__":
    unittest.main()
