import os
import tempfile
import unittest
from src.utils import count_tokens, format_tokens, export_to_markdown, export_to_json, save_output_to_file, check_file_exists
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay

class TestUtils(unittest.TestCase):
    def test_token_counting(self):
        text = "Engineering thermodynamics and fluid mechanics"
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertEqual(format_tokens(text), count)

    def test_file_exists(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.assertTrue(check_file_exists(tmp.name))
        self.assertFalse(check_file_exists("/nonexistent/file/path"))

    def test_export_to_markdown(self):
        summary = SummaryResponse(topics=[TopicSummary(topic="Physics", summary="Newton's laws.")])
        md = export_to_markdown(summary)
        self.assertIn("# Topic-Wise Summary", md)
        self.assertIn("## Physics", md)

        mcqs = MCQList(questions=[
            MCQ(
                question="What is F?",
                options=["m*a", "m/a", "m*v", "m*g"],
                correct_answer="m*a",
                explanation="Newton's second law"
            )
        ])
        md_mcq = export_to_markdown(mcqs)
        self.assertIn("# Practice MCQs", md_mcq)
        self.assertIn("Newton's second law", md_mcq)

        plan = RevisionPlan(plan=[
            RevisionDay(day=1, topics=["Math"], activities=["Review calculus"], estimated_time_hours=2.0)
        ])
        md_plan = export_to_markdown(plan)
        self.assertIn("# 7-Day Revision Plan", md_plan)
        self.assertIn("Day 1", md_plan)

    def test_export_to_json(self):
        summary = SummaryResponse(topics=[TopicSummary(topic="Physics", summary="Newton's laws.")])
        json_str = export_to_json(summary)
        self.assertIn('"topic": "Physics"', json_str)

    def test_save_output_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "subdir", "output.txt")
            save_output_to_file("Hello output", out_file)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "Hello output")

if __name__ == "__main__":
    unittest.main()
