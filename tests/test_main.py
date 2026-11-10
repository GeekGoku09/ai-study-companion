import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.main import (
    format_mcqs_markdown,
    format_output,
    format_plan_markdown,
    format_summary_markdown,
    parse_arguments,
    run_pipeline,
)
from src.schemas import (
    MCQ,
    DocumentChunk,
    MCQList,
    RevisionDay,
    RevisionPlan,
    SummaryResponse,
    TopicSummary,
)


class TestMain(unittest.TestCase):
    def test_parse_arguments(self):
        args = parse_arguments(["--pdf", "test.pdf", "--mode", "summary", "--format", "json", "--model", "gpt-4o", "--timeout", "30.0"])
        self.assertEqual(args.pdf, "test.pdf")
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.format, "json")
        self.assertEqual(args.model, "gpt-4o")
        self.assertEqual(args.timeout, 30.0)

    def test_format_summary_markdown(self):
        summary = SummaryResponse(topics=[TopicSummary(topic="Math", summary="Math summary")])
        md = format_summary_markdown(summary)
        self.assertIn("# Topic-Wise Summary", md)
        self.assertIn("## Math", md)

    def test_format_mcqs_markdown(self):
        mcqs = MCQList(questions=[MCQ(question="Q?", options=["A", "B", "C", "D"], correct_answer="A", explanation="Exp")])
        md = format_mcqs_markdown(mcqs)
        self.assertIn("# Exam-Style MCQs", md)
        self.assertIn("### Q1: Q?", md)

    def test_format_plan_markdown(self):
        plan = RevisionPlan(plan=[RevisionDay(day=1, topics=["Topic 1"], activities=["Act 1"], estimated_time_hours=2.0)])
        md = format_plan_markdown(plan)
        self.assertIn("# 7-Day Revision Plan", md)
        self.assertIn("## Day 1", md)

    def test_format_output_json(self):
        summary = SummaryResponse(topics=[TopicSummary(topic="Physics", summary="Kinematics")])
        json_out = format_output(summary, "summary", "json")
        data = json.loads(json_out)
        self.assertEqual(data["topics"][0]["topic"], "Physics")

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.check_file_exists")
    def test_run_pipeline_summary_with_output_file(self, mock_exists, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_exists.return_value = True
        mock_ingest.return_value = [DocumentChunk(content="Text", metadata={"page": 1})]

        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [DocumentChunk(content="Text", metadata={"page": 1})]

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Test Topic", summary="Test Summary")]
        )

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            output = run_pipeline(
                pdf_path="dummy.pdf",
                mode="summary",
                output_format="markdown",
                output_path=tmp_path,
            )
            self.assertIn("# Topic-Wise Summary", output)
            self.assertIn("Test Topic", output)

            with open(tmp_path, encoding="utf-8") as f:
                saved_content = f.read()
            self.assertEqual(output, saved_content)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
