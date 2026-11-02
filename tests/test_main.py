import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from src.main import main
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay, DocumentChunk

class TestMainCLI(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_exits(self):
        with patch("sys.argv", ["main.py", "--mode", "summary", "--pdf", "dummy.pdf"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)

    @patch("src.main.ingest_pdf")
    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.Summarizer")
    def test_main_summary_flow(self, mock_summarizer_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_ingest.return_value = [DocumentChunk(content="content", metadata={"page": 1})]
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [DocumentChunk(content="content", metadata={"page": 1})]

        mock_summarizer = mock_summarizer_cls.return_value
        mock_summarizer.generate_summary.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Test Topic", summary="Test Summary")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "summary.md")
            save_dir = os.path.join(tmpdir, "index")
            test_argv = [
                "main.py",
                "--pdf", "sample.pdf",
                "--mode", "summary",
                "--save-index", save_dir,
                "--output", out_file,
                "--verbose"
            ]
            with patch("sys.argv", test_argv), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                main()

            mock_retriever.save_index.assert_called_once_with(save_dir)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file) as f:
                content = f.read()
                self.assertIn("Test Topic", content)

    @patch("src.main.Retriever")
    @patch("src.main.LLMService")
    @patch("src.main.MCQGenerator")
    def test_main_mcq_flow_with_load_index(self, mock_mcq_cls, mock_llm_cls, mock_retriever_cls):
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [DocumentChunk(content="content", metadata={"page": 1})]

        mock_mcq = mock_mcq_cls.return_value
        mock_mcq.generate_mcqs.return_value = MCQList(
            questions=[
                MCQ(
                    question="What is X?",
                    options=["A", "B", "C", "D"],
                    correct_answer="A",
                    explanation="Reason"
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "mcqs.json")
            load_dir = os.path.join(tmpdir, "index")
            test_argv = [
                "main.py",
                "--load-index", load_dir,
                "--mode", "mcq",
                "--num-mcq", "1",
                "--output", out_file
            ]
            with patch("sys.argv", test_argv), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                main()

            mock_retriever.load_index.assert_called_once_with(load_dir)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file) as f:
                content = f.read()
                self.assertIn("What is X?", content)

if __name__ == "__main__":
    unittest.main()
