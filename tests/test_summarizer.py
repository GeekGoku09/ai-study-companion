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
        expected = SummaryResponse(topics=[
            TopicSummary(topic="Thermodynamics", summary="Study of heat and work."),
            TopicSummary(topic="Fluid Mechanics", summary="Study of fluids at rest and in motion.")
        ])
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = expected

        summarizer = Summarizer(mock_llm)
        chunks = [
            DocumentChunk(content="Thermodynamics is about energy.", metadata={"page": 1}),
            DocumentChunk(content="Fluids involve Navier-Stokes equations.", metadata={"page": 2})
        ]
        result = summarizer.generate_summary(chunks)

        self.assertEqual(len(result.topics), 2)
        self.assertEqual(result.topics[0].topic, "Thermodynamics")
        mock_llm.call_structured.assert_called_once()

if __name__ == "__main__":
    unittest.main()
