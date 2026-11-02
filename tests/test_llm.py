import os
import unittest
from unittest.mock import MagicMock, patch
from openai import RateLimitError, APIError
from src.llm import LLMService
from src.schemas import SummaryResponse, TopicSummary

class TestLLMService(unittest.TestCase):
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            with self.assertRaises(ValueError):
                LLMService(api_key=None)

    @patch("src.llm.OpenAI")
    def test_call_structured_success(self, mock_openai):
        mock_client = mock_openai.return_value
        expected = SummaryResponse(topics=[TopicSummary(topic="Topic A", summary="Summary A")])
        mock_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=expected, refusal=None))]
        )

        service = LLMService(api_key="fake-key")
        result = service.call_structured("sys", "user", SummaryResponse)
        self.assertEqual(result, expected)

    @patch("src.llm.OpenAI")
    def test_call_structured_retry_on_rate_limit(self, mock_openai):
        mock_client = mock_openai.return_value
        expected = SummaryResponse(topics=[TopicSummary(topic="Topic A", summary="Summary A")])
        success_response = MagicMock(choices=[MagicMock(message=MagicMock(parsed=expected, refusal=None))])

        mock_client.beta.chat.completions.parse.side_effect = [
            RateLimitError("Rate limit exceeded", response=MagicMock(status_code=429), body=None),
            success_response
        ]

        service = LLMService(api_key="fake-key", max_retries=2, retry_delay=0.01)
        result = service.call_structured("sys", "user", SummaryResponse)
        self.assertEqual(result, expected)
        self.assertEqual(mock_client.beta.chat.completions.parse.call_count, 2)

    @patch("src.llm.OpenAI")
    def test_call_structured_max_retries_exceeded(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.beta.chat.completions.parse.side_effect = RateLimitError(
            "Rate limit exceeded", response=MagicMock(status_code=429), body=None
        )

        service = LLMService(api_key="fake-key", max_retries=1, retry_delay=0.01)
        with self.assertRaises(RuntimeError):
            service.call_structured("sys", "user", SummaryResponse)

    @patch("src.llm.OpenAI")
    def test_call_structured_refusal(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=None, refusal="Policy violation"))]
        )

        service = LLMService(api_key="fake-key")
        with self.assertRaises(ValueError):
            service.call_structured("sys", "user", SummaryResponse)

    def test_prompt_helpers(self):
        service = LLMService(api_key="fake-key")
        sys_p, user_p = service.get_summarization_prompt("Context 1")
        self.assertIn("academic assistant", sys_p)
        self.assertIn("Context 1", user_p)

        sys_m, user_m = service.get_mcq_prompt("Context 2", num_questions=7)
        self.assertIn("examiner", sys_m)
        self.assertIn("7 MCQs", user_m)

        sys_pl, user_pl = service.get_planner_prompt(["Topic 1", "Topic 2"])
        self.assertIn("study coach", sys_pl)
        self.assertIn("Topic 1, Topic 2", user_pl)

if __name__ == "__main__":
    unittest.main()
