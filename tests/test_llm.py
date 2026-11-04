import unittest
from unittest.mock import MagicMock, patch
from openai import RateLimitError
from src.llm import LLMService
from src.schemas import TopicSummary, SummaryResponse

class TestLLMService(unittest.TestCase):
    @patch('src.llm.OpenAI')
    def test_call_structured_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_parsed = SummaryResponse(topics=[
            TopicSummary(topic="Statics", summary="Force balance and vectors.")
        ])
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="fake-key")
        result = service.call_structured("system", "user", SummaryResponse)

        self.assertEqual(len(result.topics), 1)
        self.assertEqual(result.topics[0].topic, "Statics")

    @patch('src.llm.time.sleep')
    @patch('src.llm.OpenAI')
    def test_call_structured_retry_on_rate_limit(self, mock_openai_cls, mock_sleep):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_parsed = SummaryResponse(topics=[TopicSummary(topic="Topic A", summary="Summary text with sufficient length.")])
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Fail once with RateLimitError, then succeed
        mock_client.beta.chat.completions.parse.side_effect = [
            RateLimitError("Rate limit reached", response=MagicMock(status_code=429), body=None),
            mock_response
        ]

        service = LLMService(api_key="fake-key", max_retries=2, base_delay=0.01)
        result = service.call_structured("system", "user", SummaryResponse)

        self.assertEqual(len(result.topics), 1)
        self.assertEqual(mock_client.beta.chat.completions.parse.call_count, 2)
        mock_sleep.assert_called_once()

    def test_missing_api_key_raises_error(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                LLMService(api_key=None)

    @patch('src.llm.OpenAI')
    def test_prompt_generation(self, mock_openai_cls):
        service = LLMService(api_key="fake-key")
        sys_p, user_p = service.get_summarization_prompt("test context")
        self.assertIn("<<DOCUMENT_CONTEXT>>", user_p)
        self.assertIn("test context", user_p)

        sys_m, user_m = service.get_mcq_prompt("formula context", num_questions=3)
        self.assertIn("3 high-quality MCQs", user_m)
        self.assertIn("<<DOCUMENT_CONTEXT>>", user_m)

        sys_pl, user_pl = service.get_planner_prompt(["Math", "Physics"])
        self.assertIn("Math", user_pl)
        self.assertIn("Physics", user_pl)

if __name__ == "__main__":
    unittest.main()
