import unittest
from unittest.mock import MagicMock
from src.summarizer import Summarizer
from src.llm import LLMService
from src.schemas import SummaryResponse, TopicSummary, DocumentChunk

class TestSummarizer(unittest.TestCase):
    def test_generate_summary_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)
        result = summarizer.generate_summary([])
        self.assertIsInstance(result, SummaryResponse)
        self.assertEqual(len(result.topics), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_summary_success(self):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.get_summarization_prompt.return_value = ("system prompt", "user prompt")
        expected_response = SummaryResponse(topics=[
            TopicSummary(topic="Thermodynamics", summary="First and second laws of thermodynamics.")
        ])
        mock_llm.call_structured.return_value = expected_response

        summarizer = Summarizer(mock_llm)
        chunks = [
            DocumentChunk(content="Thermodynamics is the study of energy transfer.", metadata={"page": 1}),
            DocumentChunk(content="Entropy always increases.", metadata={"page": 2})
        ]

        result = summarizer.generate_summary(chunks)
        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Thermodynamics")
        mock_llm.call_structured.assert_called_once()

if __name__ == "__main__":
    unittest.main()
