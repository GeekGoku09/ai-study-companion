import unittest
from unittest.mock import MagicMock
from src.summarizer import Summarizer
from src.llm import LLMService
from src.schemas import DocumentChunk, SummaryResponse, TopicSummary

class TestSummarizer(unittest.TestCase):
    def test_generate_summary_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)
        result = summarizer.generate_summary([])
        self.assertEqual(len(result.topics), 0)
        self.assertIsInstance(result, SummaryResponse)

    def test_generate_summary_valid(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_response = SummaryResponse(topics=[
            TopicSummary(topic="Calculus", summary="Derivatives and Integrals of functions.")
        ])
        mock_llm.get_summarization_prompt.return_value = ("sys prompt", "user prompt")
        mock_llm.call_structured.return_value = expected_response

        summarizer = Summarizer(mock_llm)
        chunks = [DocumentChunk(content="Calculus explains change.", metadata={"page": 1})]
        result = summarizer.generate_summary(chunks)

        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Calculus")
        md = result.to_markdown()
        self.assertIn("## Calculus", md)

if __name__ == "__main__":
    unittest.main()
