import unittest
from unittest.mock import MagicMock
from src.summarizer import Summarizer
from src.llm import LLMService
from src.schemas import DocumentChunk, SummaryResponse, TopicSummary

class TestSummarizer(unittest.TestCase):
    def test_generate_summary_success(self):
        mock_llm = MagicMock(spec=LLMService)
        expected = SummaryResponse(topics=[
            TopicSummary(topic="Thermodynamics", summary="First and second laws.")
        ])
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected

        summarizer = Summarizer(mock_llm)
        chunks = [
            DocumentChunk(content="Thermodynamics deals with heat and work.", metadata={"page": 1})
        ]

        result = summarizer.generate_summary(chunks)
        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Thermodynamics")
        mock_llm.call_structured.assert_called_once()

    def test_generate_summary_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)

        result = summarizer.generate_summary([])
        self.assertEqual(len(result.topics), 0)
        mock_llm.call_structured.assert_not_called()

if __name__ == "__main__":
    unittest.main()
