import unittest
from unittest.mock import MagicMock, patch
from src.llm import LLMService
from src.schemas import SummaryResponse
from src.exceptions import LLMServiceError

class TestLLMService(unittest.TestCase):
    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_init_missing_key(self, mock_getenv, mock_openai):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            LLMService(api_key=None)

    @patch('src.llm.OpenAI')
    def test_call_structured_success(self, mock_openai):
        mock_client = mock_openai.return_value
        expected_parsed = SummaryResponse(topics=[])

        mock_choice = MagicMock()
        mock_choice.message.parsed = expected_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="fake-key")
        result = service.call_structured("sys", "user", SummaryResponse)
        self.assertEqual(result, expected_parsed)

    @patch('src.llm.OpenAI')
    def test_call_structured_error_raised(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.beta.chat.completions.parse.side_effect = RuntimeError("API down")

        service = LLMService(api_key="fake-key")
        with self.assertRaises(LLMServiceError):
            service.call_structured("sys", "user", SummaryResponse)

if __name__ == "__main__":
    unittest.main()
