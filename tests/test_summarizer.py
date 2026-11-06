import unittest
from unittest.mock import MagicMock
from src.summarizer import Summarizer
from src.llm import LLMService
from src.schemas import SummaryResponse, TopicSummary, DocumentChunk

class TestSummarizer(unittest.TestCase):
    def test_generate_summary_success(self):
        mock_llm = MagicMock(spec=LLMService)
        expected_summary = SummaryResponse(
            topics=[
                TopicSummary(topic="Thermodynamics", summary="Study of heat and work."),
                TopicSummary(topic="Fluid Mechanics", summary="Study of fluids in motion.")
            ]
        )
        mock_llm.get_summarization_prompt.return_value = ("system", "user")
        mock_llm.call_structured.return_value = expected_summary

        summarizer = Summarizer(mock_llm)
        chunks = [
            DocumentChunk(content="Thermodynamics notes", metadata={"page": 1}),
            DocumentChunk(content="Fluid mechanics notes", metadata={"page": 2})
        ]

        result = summarizer.generate_summary(chunks)
        self.assertIsInstance(result, SummaryResponse)
        self.assertEqual(len(result.topics), 2)
        self.assertEqual(result.topics[0].topic, "Thermodynamics")

    def test_generate_summary_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        summarizer = Summarizer(mock_llm)
        result = summarizer.generate_summary([])
        self.assertEqual(len(result.topics), 0)

if __name__ == '__main__':
    unittest.main()
