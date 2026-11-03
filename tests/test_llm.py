import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from openai import RateLimitError
from src.llm import LLMService, LLMServiceError

class SampleResponse(BaseModel):
    message: str

class TestLLMService(unittest.TestCase):
    @patch('src.llm.OpenAI')
    def test_call_structured_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_parsed = SampleResponse(message="Success")
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="test-key")
        result = service.call_structured("sys", "user", SampleResponse)

        self.assertEqual(result.message, "Success")

    @patch('src.llm.OpenAI')
    def test_call_structured_refusal(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.parsed = None
        mock_choice.message.refusal = "Content not allowed"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="test-key", max_retries=1)
        with self.assertRaises(LLMServiceError):
            service.call_structured("sys", "user", SampleResponse)

    @patch('src.llm.time.sleep')
    @patch('src.llm.OpenAI')
    def test_call_structured_transient_retry(self, mock_openai, mock_sleep):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_parsed = SampleResponse(message="Retry succeeded")
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # First call raises RateLimitError, second succeeds
        dummy_response = MagicMock()
        dummy_response.status_code = 429
        mock_client.beta.chat.completions.parse.side_effect = [
            RateLimitError("Rate limit", response=dummy_response, body=None),
            mock_response
        ]

        service = LLMService(api_key="test-key", max_retries=2)
        result = service.call_structured("sys", "user", SampleResponse)

        self.assertEqual(result.message, "Retry succeeded")
        mock_sleep.assert_called_once()

    def test_missing_api_key_raises(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                LLMService(api_key=None)

if __name__ == "__main__":
    unittest.main()
