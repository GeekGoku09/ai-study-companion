import unittest
import os
import tempfile
import json
from src.utils import check_file_exists, count_tokens, format_tokens, export_to_markdown, export_to_json
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay

class TestUtils(unittest.TestCase):
    def test_token_counting(self):
        text = "Hello world from the engineering companion."
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertEqual(format_tokens(text), count)

    def test_check_file_exists(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.assertTrue(check_file_exists(tmp.name))
        self.assertFalse(check_file_exists("/nonexistent/file/path/here.txt"))

    def test_export_to_markdown_summary(self):
        summary = SummaryResponse(topics=[
            TopicSummary(topic="Fluid Dynamics", summary="Navier-Stokes equations describe viscous fluid substances.")
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "summary.md")
            export_to_markdown(summary, out_file)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r") as f:
                content = f.read()
            self.assertIn("# Topic-Wise Study Summary", content)
            self.assertIn("## Fluid Dynamics", content)
            self.assertIn("Navier-Stokes", content)

    def test_export_to_markdown_mcqs(self):
        mcqs = MCQList(questions=[
            MCQ(
                question="What is the unit of force?",
                options=["Joule", "Watt", "Newton", "Pascal"],
                correct_answer="Newton",
                explanation="1 Newton = 1 kg*m/s^2."
            )
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "mcqs.md")
            export_to_markdown(mcqs, out_file)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r") as f:
                content = f.read()
            self.assertIn("# Practice Examination MCQs", content)
            self.assertIn("What is the unit of force?", content)
            self.assertIn("Newton", content)

    def test_export_to_json(self):
        plan = RevisionPlan(plan=[
            RevisionDay(day=1, topics=["Circuits"], activities=["Review Ohm's law"], estimated_time_hours=2.0)
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "plan.json")
            export_to_json(plan, out_file)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r") as f:
                data = json.load(f)
            self.assertIn("plan", data)
            self.assertEqual(data["plan"][0]["day"], 1)

if __name__ == '__main__':
    unittest.main()
