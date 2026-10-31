import unittest
from unittest.mock import MagicMock, patch
from src.llm import LLMService
from src.schemas import SummaryResponse, TopicSummary

class TestLLMService(unittest.TestCase):
    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_truncate_context_to_budget(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        llm = LLMService(api_key="fake-key", max_context_tokens=10)
        long_text = "word " * 100
        truncated = llm.truncate_context_to_budget(long_text)
        self.assertIn("[Context truncated to adhere to token budget]", truncated)

    @patch('src.llm.OpenAI')
    def test_call_structured_success(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_parsed = SummaryResponse(topics=[TopicSummary(topic="T1", summary="S1 valid")])
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        llm = LLMService(api_key="fake-key")
        result = llm.call_structured("system", "user", SummaryResponse)
        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "T1")

if __name__ == "__main__":
    unittest.main()
