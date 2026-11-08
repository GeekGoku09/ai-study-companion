import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.main import create_parser, run_pipeline
from src.schemas import (
    MCQ,
    MCQList,
    RevisionDay,
    RevisionPlan,
    SummaryResponse,
    TopicSummary,
)


class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_parser_arguments(self):
        args = self.parser.parse_args([
            "--pdf", "sample.pdf",
            "--mode", "summary",
            "--output", "out.md",
            "--format", "markdown",
            "--model", "gpt-4o",
            "--timeout", "45",
        ])
        self.assertEqual(args.pdf, "sample.pdf")
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.output, "out.md")
        self.assertEqual(args.format, "markdown")
        self.assertEqual(args.model, "gpt-4o")
        self.assertEqual(args.timeout, 45.0)

    def test_missing_api_key_returns_code_1(self):
        with patch.dict("os.environ", {}, clear=True):
            args = self.parser.parse_args(["--pdf", "test.pdf", "--mode", "summary"])
            exit_code = run_pipeline(args)
            self.assertEqual(exit_code, 1)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_invalid_pdf_path_returns_code_1(self):
        args = self.parser.parse_args(["--pdf", "nonexistent.pdf", "--mode", "summary"])
        exit_code = run_pipeline(args)
        self.assertEqual(exit_code, 1)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.main.validate_pdf_path")
    @patch("src.main.ingest_pdf")
    def test_empty_chunks_returns_code_1(self, mock_ingest, _mock_validate):
        mock_ingest.return_value = []
        args = self.parser.parse_args(["--pdf", "dummy.pdf", "--mode", "summary"])
        exit_code = run_pipeline(args)
        self.assertEqual(exit_code, 1)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.main.validate_pdf_path")
    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    def test_run_pipeline_summary_markdown_with_output_file(
        self,
        mock_summarizer_cls,
        _mock_llm_cls,
        mock_retriever_cls,
        mock_ingest,
        _mock_validate,
    ):
        mock_chunk = MagicMock(content="Sample content", metadata={"page": 1})
        mock_ingest.return_value = [mock_chunk]

        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [mock_chunk]

        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Physics", summary="Newtonian Mechanics")]
        )

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            args = self.parser.parse_args([
                "--pdf", "dummy.pdf",
                "--mode", "summary",
                "--output", output_path,
                "--format", "markdown",
            ])
            exit_code = run_pipeline(args)
            self.assertEqual(exit_code, 0)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# Topic-Wise Summary", content)
            self.assertIn("## Physics", content)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.main.validate_pdf_path")
    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.MCQGenerator")
    def test_run_pipeline_mcq_json(
        self,
        mock_mcq_cls,
        _mock_llm_cls,
        mock_retriever_cls,
        mock_ingest,
        _mock_validate,
    ):
        mock_chunk = MagicMock(content="Sample content", metadata={"page": 1})
        mock_ingest.return_value = [mock_chunk]

        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [mock_chunk]

        mock_mcq_gen = mock_mcq_cls.return_value
        mock_mcq_gen.generate_mcqs.return_value = MCQList(
            questions=[
                MCQ(
                    question="What is F?",
                    options=["m*a", "m/a", "a/m", "0"],
                    correct_answer="m*a",
                    explanation="Newton's 2nd law",
                )
            ]
        )

        args = self.parser.parse_args([
            "--pdf", "dummy.pdf",
            "--mode", "mcq",
            "--num-mcq", "1",
            "--format", "json",
        ])
        exit_code = run_pipeline(args)
        self.assertEqual(exit_code, 0)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.main.validate_pdf_path")
    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    @patch("src.main.RevisionPlanner")
    def test_run_pipeline_plan(
        self,
        mock_planner_cls,
        mock_summarizer_cls,
        _mock_llm_cls,
        mock_retriever_cls,
        mock_ingest,
        _mock_validate,
    ):
        mock_chunk = MagicMock(content="Sample content", metadata={"page": 1})
        mock_ingest.return_value = [mock_chunk]

        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [mock_chunk]

        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Algorithms", summary="Sorting")]
        )

        mock_planner = mock_planner_cls.return_value
        mock_planner.generate_plan.return_value = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Algorithms"],
                    activities=["Merge Sort Implementation"],
                    estimated_time_hours=2.0,
                )
            ]
        )

        args = self.parser.parse_args([
            "--pdf", "dummy.pdf",
            "--mode", "plan",
        ])
        exit_code = run_pipeline(args)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
