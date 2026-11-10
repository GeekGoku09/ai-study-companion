import unittest
from unittest.mock import MagicMock, patch

from openai import RateLimitError

from src.llm import LLMService
from src.schemas import TopicSummary


class TestLLMService(unittest.TestCase):
    def test_init_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            LLMService(api_key=None)

    @patch("src.llm.OpenAI")
    def test_call_structured_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        expected_obj = TopicSummary(topic="Test", summary="Summary text")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = expected_obj
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="test-key")
        result = service.call_structured("sys prompt", "user prompt", TopicSummary)

        self.assertEqual(result.topic, "Test")
        self.assertEqual(result.summary, "Summary text")

    @patch("src.llm.OpenAI")
    @patch("time.sleep")
    def test_call_structured_retry_on_rate_limit(self, mock_sleep, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        expected_obj = TopicSummary(topic="Test", summary="Summary text")
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.parsed = expected_obj

        # Fail once with RateLimitError, then succeed
        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_err = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )
        mock_client.beta.chat.completions.parse.side_effect = [rate_limit_err, success_response]

        service = LLMService(api_key="test-key", max_retries=1)
        result = service.call_structured("sys prompt", "user prompt", TopicSummary)

        self.assertEqual(result.topic, "Test")
        self.assertEqual(mock_client.beta.chat.completions.parse.call_count, 2)
        mock_sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
