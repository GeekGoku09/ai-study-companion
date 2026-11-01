import unittest
import tempfile
import os
import json
from unittest.mock import MagicMock, patch
from src.main import run_pipeline, format_output_markdown, save_output
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay, DocumentChunk

class TestMainCLI(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_run_pipeline_missing_api_key(self):
        with self.assertRaises(ValueError) as ctx:
            run_pipeline(pdf_path="any.pdf", mode="summary")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_run_pipeline_missing_pdf_without_load_index(self):
        with self.assertRaises(ValueError) as ctx:
            run_pipeline(pdf_path=None, mode="summary")
        self.assertIn("Must provide --pdf", str(ctx.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.main.ingest_pdf')
    @patch('src.main.Retriever')
    @patch('src.main.LLMService')
    def test_run_pipeline_summary_flow(self, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_chunk = DocumentChunk(content="Operating systems manage hardware.", metadata={"page": 1})
        mock_ingest.return_value = [mock_chunk]

        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.chunks = [mock_chunk]
        mock_retriever.retrieve.return_value = [mock_chunk]

        mock_llm = mock_llm_cls.return_value
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = SummaryResponse(
            topics=[TopicSummary(topic="OS Concepts", summary="Kernel and process management.")]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "summary.md")
            result = run_pipeline(
                pdf_path="test.pdf",
                mode="summary",
                output_path=out_file
            )
            self.assertEqual(len(result.topics), 1)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file) as f:
                content = f.read()
                self.assertIn("OS Concepts", content)

    def test_format_output_markdown(self):
        mcq_result = MCQList(questions=[
            MCQ(
                question="What is RAM?",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                explanation="Random Access Memory"
            )
        ])
        md = format_output_markdown("mcq", mcq_result)
        self.assertIn("### Q1: What is RAM?", md)
        self.assertIn("**Correct Answer:** A", md)

if __name__ == "__main__":
    unittest.main()
