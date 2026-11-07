import argparse
import os
import tempfile
import unittest
from unittest.mock import patch

from src.main import create_parser, main, run_pipeline, validate_args
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
    def test_parser_defaults(self):
        parser = create_parser()
        args = parser.parse_args(["--pdf", "test.pdf", "--mode", "summary"])
        self.assertEqual(args.pdf, "test.pdf")
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.num_mcq, 5)
        self.assertEqual(args.format, "markdown")
        self.assertEqual(args.timeout, 60.0)

    def test_validate_args(self):
        with tempfile.NamedTemporaryFile() as tmp:
            valid_args = argparse.Namespace(pdf=tmp.name, num_mcq=5, timeout=10.0)
            # Should not raise
            validate_args(valid_args)

            invalid_mcq = argparse.Namespace(pdf=tmp.name, num_mcq=0, timeout=10.0)
            with self.assertRaises(ValueError):
                validate_args(invalid_mcq)

            invalid_timeout = argparse.Namespace(pdf=tmp.name, num_mcq=5, timeout=-1.0)
            with self.assertRaises(ValueError):
                validate_args(invalid_timeout)

        missing_pdf = argparse.Namespace(pdf="nonexistent_pdf_file.pdf", num_mcq=5, timeout=10.0)
        with self.assertRaises(FileNotFoundError):
            validate_args(missing_pdf)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    def test_run_pipeline_summary_json(self, mock_summarizer_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        with tempfile.NamedTemporaryFile() as tmp:
            mock_ingest.return_value = [DocumentChunk(content="chunk")]
            mock_retriever = mock_retriever_cls.return_value
            mock_retriever.retrieve.return_value = [DocumentChunk(content="chunk")]

            mock_summarizer = mock_summarizer_cls.return_value
            mock_summarizer.generate_summary.return_value = SummaryResponse(
                topics=[TopicSummary(topic="T1", summary="S1")]
            )

            args = argparse.Namespace(
                pdf=tmp.name,
                mode="summary",
                num_mcq=5,
                output=None,
                format="json",
                model="gpt-4o-mini",
                timeout=60.0,
            )
            output = run_pipeline(args)
            self.assertIn('"topic": "T1"', output)
            self.assertIn('"summary": "S1"', output)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.MCQGenerator")
    def test_run_pipeline_mcq_markdown(self, mock_mcq_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        with tempfile.NamedTemporaryFile() as tmp:
            mock_ingest.return_value = [DocumentChunk(content="chunk")]
            mock_retriever = mock_retriever_cls.return_value
            mock_retriever.retrieve.return_value = [DocumentChunk(content="chunk")]

            mock_mcq = mock_mcq_cls.return_value
            mock_mcq.generate_mcqs.return_value = MCQList(
                questions=[
                    MCQ(
                        question="Sample Q?",
                        options=["A", "B", "C", "D"],
                        correct_answer="A",
                        explanation="Because A.",
                    )
                ]
            )

            args = argparse.Namespace(
                pdf=tmp.name,
                mode="mcq",
                num_mcq=1,
                output=None,
                format="markdown",
                model="gpt-4o-mini",
                timeout=60.0,
            )
            output = run_pipeline(args)
            self.assertIn("### Q1: Sample Q?", output)
            self.assertIn("- **A)** A", output)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    @patch("src.main.RevisionPlanner")
    def test_run_pipeline_plan_markdown(
        self, mock_planner_cls, mock_summarizer_cls, mock_llm_cls, mock_retriever_cls, mock_ingest
    ):
        with tempfile.NamedTemporaryFile() as tmp:
            mock_ingest.return_value = [DocumentChunk(content="chunk")]
            mock_retriever = mock_retriever_cls.return_value
            mock_retriever.retrieve.return_value = [DocumentChunk(content="chunk")]

            mock_summarizer = mock_summarizer_cls.return_value
            mock_summarizer.generate_summary.return_value = SummaryResponse(
                topics=[TopicSummary(topic="Robotics", summary="Robotics notes")]
            )

            mock_planner = mock_planner_cls.return_value
            mock_planner.generate_plan.return_value = RevisionPlan(
                plan=[
                    RevisionDay(
                        day=1,
                        topics=["Robotics"],
                        activities=["Study inverse kinematics"],
                        estimated_time_hours=3.0,
                    )
                ]
            )

            args = argparse.Namespace(
                pdf=tmp.name,
                mode="plan",
                num_mcq=5,
                output=None,
                format="markdown",
                model="gpt-4o-mini",
                timeout=60.0,
            )
            output = run_pipeline(args)
            self.assertIn("## Day 1", output)
            self.assertIn("Robotics", output)

    @patch("src.main.run_pipeline")
    @patch("os.getenv")
    def test_main_cli_execution_with_output_file(self, mock_getenv, mock_run_pipeline):
        mock_getenv.return_value = "fake-key"
        mock_run_pipeline.return_value = "# Output Content"

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_tmp,
            patch("sys.argv", ["main.py", "--pdf", pdf_tmp.name, "--mode", "summary", "--output", pdf_tmp.name + ".out.md"]),
        ):
            out_file = pdf_tmp.name + ".out.md"
            main()
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "# Output Content")
            os.remove(out_file)

    @patch("os.getenv")
    def test_main_missing_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_tmp,
            patch("sys.argv", ["main.py", "--pdf", pdf_tmp.name, "--mode", "summary"]),
            self.assertRaises(SystemExit) as ctx,
        ):
            main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
