import os
import unittest
from unittest.mock import patch
from src.main import build_arg_parser, format_output, run_app
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay


class TestMain(unittest.TestCase):
    def test_build_arg_parser_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--pdf", "sample.pdf", "--mode", "summary"])
        self.assertEqual(args.pdf, "sample.pdf")
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.num_mcq, 5)
        self.assertEqual(args.format, "markdown")
        self.assertEqual(args.model, "gpt-4o-mini")
        self.assertEqual(args.timeout, 60.0)

    def test_format_output(self):
        summary = SummaryResponse(topics=[TopicSummary(topic="Math", summary="Algebra")])
        md_out = format_output(summary, "summary", "markdown")
        self.assertIn("## Math", md_out)

        json_out = format_output(summary, "summary", "json")
        self.assertIn('"topic": "Math"', json_out)

    @patch.dict(os.environ, {}, clear=True)
    def test_run_app_missing_api_key(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--pdf", "test.pdf", "--mode", "summary"])
        exit_code = run_app(args)
        self.assertEqual(exit_code, 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    def test_run_app_invalid_num_mcq(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--pdf", "test.pdf", "--mode", "mcq", "--num-mcq", "0"])
        exit_code = run_app(args)
        self.assertEqual(exit_code, 1)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    def test_run_app_summary_mode(self, mock_summarizer_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_ingest.return_value = ["dummy_chunk"]
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = ["dummy_chunk"]

        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Circuits", summary="Ohm's Law")]
        )

        parser = build_arg_parser()
        args = parser.parse_args(["--pdf", "sample.pdf", "--mode", "summary"])
        exit_code = run_app(args)
        self.assertEqual(exit_code, 0)
        mock_summarizer.generate_summary.assert_called_once()

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.MCQGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    def test_run_app_mcq_mode_output_file(self, mock_mcq_gen_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_ingest.return_value = ["dummy_chunk"]
        mock_mcq_gen = mock_mcq_gen_cls.return_value
        mock_mcq_gen.generate_mcqs.return_value = MCQList(
            questions=[
                MCQ(
                    question="What is V=IR?",
                    options=["Newton's Law", "Ohm's Law", "Hooke's Law", "Boyle's Law"],
                    correct_answer="Ohm's Law",
                    explanation="V=IR represents Ohm's Law.",
                )
            ]
        )

        test_out_path = "tests/test_output.md"
        try:
            parser = build_arg_parser()
            args = parser.parse_args([
                "--pdf", "sample.pdf",
                "--mode", "mcq",
                "--output", test_out_path,
                "--format", "markdown"
            ])
            exit_code = run_app(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(test_out_path))
            with open(test_out_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Ohm's Law", content)
        finally:
            if os.path.exists(test_out_path):
                os.remove(test_out_path)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    @patch("src.main.RevisionPlanner")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    def test_run_app_plan_mode(self, mock_planner_cls, mock_summarizer_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_ingest.return_value = ["dummy_chunk"]
        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Dynamics", summary="Forces")]
        )
        mock_planner = mock_planner_cls.return_value
        mock_planner.generate_plan.return_value = RevisionPlan(
            plan=[
                RevisionDay(
                    day=1,
                    topics=["Dynamics"],
                    activities=["Review Newton's second law"],
                    estimated_time_hours=2.0,
                )
            ]
        )

        parser = build_arg_parser()
        args = parser.parse_args(["--pdf", "sample.pdf", "--mode", "plan", "--format", "json"])
        exit_code = run_app(args)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
