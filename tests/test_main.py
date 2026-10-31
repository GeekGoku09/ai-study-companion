import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import argparse
from src.main import run_app, build_parser
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay, DocumentChunk

class TestMain(unittest.TestCase):
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--mode", "summary", "--pdf", "sample.pdf"])
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.num_mcq, 5)
        self.assertEqual(args.format, "markdown")

    @patch('os.getenv')
    def test_run_app_missing_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        args = argparse.Namespace(
            pdf="test.pdf",
            mode="summary",
            num_mcq=5,
            save_index=None,
            load_index=None,
            output=None,
            format="markdown",
            model=None
        )
        code = run_app(args)
        self.assertEqual(code, 1)

    @patch('os.getenv')
    def test_run_app_missing_pdf_and_load_index(self, mock_getenv):
        mock_getenv.return_value = "fake-key"
        args = argparse.Namespace(
            pdf=None,
            mode="summary",
            num_mcq=5,
            save_index=None,
            load_index=None,
            output=None,
            format="markdown",
            model=None
        )
        code = run_app(args)
        self.assertEqual(code, 1)

    @patch('src.main.Retriever')
    @patch('src.main.ingest_pdf')
    @patch('src.main.Summarizer')
    @patch('os.getenv')
    def test_run_app_summary_success_with_output(self, mock_getenv, mock_summarizer_cls, mock_ingest, mock_retriever_cls):
        mock_getenv.return_value = "fake-key"
        mock_ingest.return_value = [DocumentChunk(content="Physics notes content", metadata={"page": 1})]

        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = mock_ingest.return_value

        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Physics", summary="Kinematics and Dynamics")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "summary.md")
            args = argparse.Namespace(
                pdf="physics.pdf",
                mode="summary",
                num_mcq=5,
                save_index=None,
                load_index=None,
                output=out_file,
                format="markdown",
                model=None
            )
            code = run_app(args)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out_file))

if __name__ == "__main__":
    unittest.main()
