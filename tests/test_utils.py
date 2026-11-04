import os
import tempfile
import unittest
from src.utils import count_tokens, format_tokens, check_file_exists, export_result
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay

class TestUtils(unittest.TestCase):
    def test_count_tokens(self):
        text = "Hello world, this is an engineering test."
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(format_tokens(text), count)

    def test_check_file_exists(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.assertTrue(check_file_exists(tmp.name))
        self.assertFalse(check_file_exists("non_existent_file_path_123.txt"))

    def test_export_result_markdown_and_json(self):
        summary = SummaryResponse(topics=[
            TopicSummary(topic="Calculus", summary="Derivatives and integrals.")
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Markdown export
            md_path = os.path.join(tmpdir, "summary.md")
            export_result(summary, md_path)
            self.assertTrue(os.path.exists(md_path))
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("# Document Summary", content)
                self.assertIn("Calculus", content)

            # JSON export
            json_path = os.path.join(tmpdir, "summary.json")
            export_result(summary, json_path)
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn('"topic": "Calculus"', content)

            # MCQ export
            mcqs = MCQList(questions=[
                MCQ(question="What is 2+2?", options=["A", "B", "C", "D"], correct_answer="A", explanation="Basic math explanation.")
            ])
            mcq_md_path = os.path.join(tmpdir, "mcqs.md")
            export_result(mcqs, mcq_md_path)
            with open(mcq_md_path, "r", encoding="utf-8") as f:
                self.assertIn("# Multiple Choice Questions", f.read())

            # Plan export
            plan = RevisionPlan(plan=[
                RevisionDay(day=1, topics=["Math"], activities=["Practice"], estimated_time_hours=2.0)
            ])
            plan_md_path = os.path.join(tmpdir, "plan.md")
            export_result(plan, plan_md_path)
            with open(plan_md_path, "r", encoding="utf-8") as f:
                self.assertIn("# 7-Day Revision Plan", f.read())

if __name__ == "__main__":
    unittest.main()
