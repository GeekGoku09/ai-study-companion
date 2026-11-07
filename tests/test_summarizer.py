import unittest
from unittest.mock import MagicMock

from src.llm import LLMService
from src.schemas import DocumentChunk, SummaryResponse, TopicSummary
from src.summarizer import Summarizer


class TestSummarizer(unittest.TestCase):
    def test_generate_summary_success(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_summary = SummaryResponse(topics=[
            TopicSummary(topic="Fluid Dynamics", summary="Incompressible Navier-Stokes equations.")
        ])
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected_summary

        summarizer = Summarizer(mock_llm)
        chunks = [
            DocumentChunk(content="Fluid flow notes", metadata={"page": 1}),
            DocumentChunk(content="More fluids", metadata={"page": 2}),
        ]

        result = summarizer.generate_summary(chunks)
        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Fluid Dynamics")
        mock_llm.call_structured.assert_called_once()

    def test_generate_summary_empty(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)
        result = summarizer.generate_summary([])
        self.assertEqual(result.topics, [])
        mock_llm.call_structured.assert_not_called()


if __name__ == "__main__":
    unittest.main()
