import unittest
from unittest.mock import MagicMock, patch
from openai import RateLimitError, APIConnectionError
from src.llm import LLMService
from src.schemas import TopicSummary

class TestLLMService(unittest.TestCase):
    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_init_missing_key_raises(self, mock_getenv, mock_openai):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            LLMService(api_key=None)

    @patch('src.llm.OpenAI')
    def test_call_structured_success(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_parsed = TopicSummary(topic="Calculus", summary="Calculus covers derivatives and integrals.")
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="test-key")
        result = service.call_structured("sys", "user", TopicSummary)
        self.assertEqual(result.topic, "Calculus")

    @patch('time.sleep')
    @patch('src.llm.OpenAI')
    def test_call_structured_retry_on_rate_limit(self, mock_openai, mock_sleep):
        mock_client = mock_openai.return_value
        mock_parsed = TopicSummary(topic="Physics", summary="Newton's laws of motion describe forces.")
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Fail first time with RateLimitError, succeed on second attempt
        rate_err = RateLimitError(
            message="Rate limit hit",
            response=MagicMock(status_code=429, headers={}),
            body=None
        )
        mock_client.beta.chat.completions.parse.side_effect = [rate_err, mock_response]

        service = LLMService(api_key="test-key", max_retries=2, backoff_factor=0.01)
        result = service.call_structured("sys", "user", TopicSummary)

        self.assertEqual(result.topic, "Physics")
        self.assertEqual(mock_client.beta.chat.completions.parse.call_count, 2)
        mock_sleep.assert_called_once()

    def test_prompts_generation(self):
        with patch('src.llm.OpenAI'):
            service = LLMService(api_key="fake-key")
            sys_s, user_s = service.get_summarization_prompt("test context")
            self.assertIn("Summarize", sys_s)
            self.assertIn("test context", user_s)

            sys_m, user_m = service.get_mcq_prompt("mcq context", num_questions=3)
            self.assertIn("MCQs", sys_m)
            self.assertIn("3 high-quality MCQs", user_m)

            sys_p, user_p = service.get_planner_prompt(["Math", "Physics"])
            self.assertIn("revision plan", sys_p)
            self.assertIn("Math, Physics", user_p)

if __name__ == '__main__':
    unittest.main()
