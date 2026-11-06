import unittest
import tempfile
import os
from unittest.mock import patch
from src.main import build_parser, run
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay, DocumentChunk
from src.utils import format_result, count_tokens

class TestMainAndUtils(unittest.TestCase):
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--pdf", "test.pdf", "--mode", "summary"])
        self.assertEqual(args.pdf, "test.pdf")
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.num_mcq, 5)
        self.assertEqual(args.format, "markdown")

    def test_format_result_markdown_and_json(self):
        summary = SummaryResponse(
            topics=[TopicSummary(topic="Algorithms", summary="Big O analysis")]
        )
        md = format_result(summary, output_format="markdown")
        self.assertIn("# Topic-wise Summary", md)
        self.assertIn("Algorithms", md)

        js = format_result(summary, output_format="json")
        self.assertIn('"topic": "Algorithms"', js)

        mcq = MCQList(
            questions=[MCQ(
                question="What is complexity of binary search?",
                options=["O(1)", "O(n)", "O(log n)", "O(n^2)"],
                correct_answer="O(log n)",
                explanation="Halves search space each step."
            )]
        )
        mcq_md = format_result(mcq, output_format="markdown")
        self.assertIn("What is complexity", mcq_md)
        self.assertIn("O(log n)", mcq_md)

        plan = RevisionPlan(
            plan=[RevisionDay(
                day=1,
                topics=["Sorting"],
                activities=["Quicksort review"],
                estimated_time_hours=2.0
            )]
        )
        plan_md = format_result(plan, output_format="markdown")
        self.assertIn("Day 1", plan_md)
        self.assertIn("Quicksort review", plan_md)

    def test_count_tokens(self):
        count = count_tokens("Hello world this is a test")
        self.assertGreater(count, 0)

    @patch('src.main.load_dotenv')
    @patch('src.main.ingest_pdf')
    @patch('src.main.Retriever')
    @patch('src.main.LLMService')
    @patch('src.main.Summarizer')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_run_summary_with_output_file(
        self, mock_getenv, mock_exists, mock_summarizer_cls, mock_llm_cls,
        mock_retriever_cls, mock_ingest, mock_dotenv
    ):
        mock_getenv.return_value = "fake-key"
        mock_exists.return_value = True
        mock_ingest.return_value = [DocumentChunk(content="chunk", metadata={"page": 1})]

        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [DocumentChunk(content="chunk", metadata={"page": 1})]

        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Test Topic", summary="Test Summary")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "summary.md")
            parser = build_parser()
            args = parser.parse_args([
                "--pdf", "sample.pdf",
                "--mode", "summary",
                "--output", out_file,
                "--format", "markdown"
            ])

            ret_code = run(args)
            self.assertEqual(ret_code, 0)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r") as f:
                content = f.read()
                self.assertIn("Test Topic", content)

    @patch('src.main.load_dotenv')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_run_validation_errors(self, mock_getenv, mock_exists, mock_dotenv):
        mock_getenv.return_value = "fake-key"
        mock_exists.return_value = False

        parser = build_parser()
        # Non-existent file
        args = parser.parse_args(["--pdf", "nonexistent.pdf", "--mode", "summary"])
        self.assertEqual(run(args), 1)

        # Invalid mcq count
        mock_exists.return_value = True
        args = parser.parse_args(["--pdf", "test.pdf", "--mode", "mcq", "--num-mcq", "0"])
        self.assertEqual(run(args), 1)

if __name__ == '__main__':
    unittest.main()
