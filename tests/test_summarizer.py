import unittest
from unittest.mock import MagicMock

from src.llm import LLMService
from src.schemas import DocumentChunk, SummaryResponse, TopicSummary
from src.summarizer import Summarizer


class TestSummarizer(unittest.TestCase):
    def test_generate_summary_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)
        result = summarizer.generate_summary([])
        self.assertIsInstance(result, SummaryResponse)
        self.assertEqual(len(result.topics), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_summary_with_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_response = SummaryResponse(
            topics=[
                TopicSummary(
                    topic="Thermodynamics",
                    summary="Study of heat and energy transfer.",
                )
            ]
        )
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected_response

        summarizer = Summarizer(mock_llm)
        chunks = [
            DocumentChunk(content="First law of thermodynamics.", metadata={"page": 1})
        ]
        result = summarizer.generate_summary(chunks)

        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Thermodynamics")
        mock_llm.call_structured.assert_called_once()


if __name__ == "__main__":
    unittest.main()
