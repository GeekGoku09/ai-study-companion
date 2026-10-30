import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from src.main import format_summary_markdown, format_mcq_markdown, format_plan_markdown, write_output_file
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay

class TestMainCLIUtilities(unittest.TestCase):
    def test_format_summary_markdown(self):
        summary = SummaryResponse(topics=[
            TopicSummary(topic="Operating Systems", summary="Manages hardware and software resources.")
        ])
        md = format_summary_markdown(summary)
        self.assertIn("# Topic-Wise Summary", md)
        self.assertIn("## Operating Systems", md)

    def test_format_mcq_markdown(self):
        mcqs = MCQList(questions=[
            MCQ(
                question="What is a process?",
                options=["Program in execution", "A disk block", "A register", "A cache"],
                correct_answer="Program in execution",
                explanation="Standard OS definition."
            )
        ])
        md = format_mcq_markdown(mcqs)
        self.assertIn("# Exam Practice MCQs", md)
        self.assertIn("Q1: What is a process?", md)
        lines = [line.strip() for line in md.splitlines()]
        self.assertTrue(any(line.startswith("- **A)** Program in execution") for line in lines))

    def test_format_plan_markdown(self):
        plan = RevisionPlan(plan=[
            RevisionDay(day=1, topics=["Math"], activities=["Practice integration"], estimated_time_hours=2.0)
        ])
        md = format_plan_markdown(plan)
        self.assertIn("# 7-Day Study Revision Plan", md)
        self.assertIn("## Day 1", md)

    def test_write_output_file_json_and_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "out.json")
            data = {"key": "value"}
            write_output_file(json_path, data, "# Markdown")
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

            md_path = os.path.join(tmpdir, "out.md")
            write_output_file(md_path, data, "# Markdown Title")
            self.assertTrue(os.path.exists(md_path))
            with open(md_path, "r") as f:
                content = f.read()
            self.assertEqual(content, "# Markdown Title")

if __name__ == "__main__":
    unittest.main()
