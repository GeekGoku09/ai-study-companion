import unittest
from unittest.mock import MagicMock
from src.summarizer import Summarizer
from src.llm import LLMService
from src.schemas import DocumentChunk, SummaryResponse, TopicSummary

class TestSummarizer(unittest.TestCase):
    def test_generate_summary_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)
        res = summarizer.generate_summary([])
        self.assertIsInstance(res, SummaryResponse)
        self.assertEqual(len(res.topics), 0)
        mock_llm.call_structured.assert_not_called()

    def test_generate_summary_with_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        expected = SummaryResponse(topics=[
            TopicSummary(topic="Thermodynamics", summary="First and second laws of thermodynamics governed by conservation of energy.")
        ])
        mock_llm.call_structured.return_value = expected

        summarizer = Summarizer(mock_llm)
        chunks = [DocumentChunk(content="Thermodynamics lecture notes.", metadata={"page": 1})]
        result = summarizer.generate_summary(chunks)

        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Thermodynamics")
        mock_llm.call_structured.assert_called_once()

    def test_summarizer_none_service_raises(self):
        with self.assertRaises(ValueError):
            Summarizer(None)

if __name__ == '__main__':
    unittest.main()
