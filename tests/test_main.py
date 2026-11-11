import tempfile
import unittest
from unittest.mock import patch
from src.main import parse_arguments, run_pipeline
from src.schemas import DocumentChunk, SummaryResponse, TopicSummary


class TestMain(unittest.TestCase):
    def test_parse_arguments_defaults(self):
        args = parse_arguments(["--pdf", "test.pdf", "--mode", "summary"])
        self.assertEqual(args.pdf, "test.pdf")
        self.assertEqual(args.mode, "summary")
        self.assertEqual(args.num_mcq, 5)
        self.assertEqual(args.format, "markdown")

    def test_run_pipeline_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                args = parse_arguments(["--pdf", "test.pdf", "--mode", "summary"])
                run_pipeline(args)
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_run_pipeline_missing_pdf(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaises(FileNotFoundError):
                args = parse_arguments(["--pdf", "non_existent_file.pdf", "--mode", "summary"])
                run_pipeline(args)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.Summarizer")
    def test_run_pipeline_summary_flow(self, mock_summarizer_cls, mock_retriever_cls, mock_ingest):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            # Set up mocks
            mock_chunk = DocumentChunk(content="Sample content for testing", metadata={"page": 1})
            mock_ingest.return_value = [mock_chunk]

            mock_retriever = mock_retriever_cls.return_value
            mock_retriever.retrieve.return_value = [mock_chunk]

            mock_summarizer = mock_summarizer_cls.return_value
            mock_summarizer.generate_summary.return_value = SummaryResponse(topics=[
                TopicSummary(topic="Test Topic", summary="Test summary content.")
            ])

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                args = parse_arguments(["--pdf", tmp_pdf.name, "--mode", "summary", "--format", "json"])
                output = run_pipeline(args)

                self.assertIn('"Test Topic"', output)
                mock_ingest.assert_called_once_with(tmp_pdf.name)
                mock_retriever.build_index.assert_called_once()
                mock_summarizer.generate_summary.assert_called_once()


if __name__ == "__main__":
    unittest.main()
