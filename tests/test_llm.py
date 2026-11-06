import unittest
from unittest.mock import MagicMock, patch
from src.llm import LLMService
from src.schemas import SummaryResponse, TopicSummary

class TestLLMService(unittest.TestCase):
    @patch('os.getenv')
    def test_missing_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            LLMService(api_key=None)

    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_call_structured(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        expected_parsed = SummaryResponse(
            topics=[TopicSummary(topic="Physics", summary="Kinematics")]
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = expected_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(model="gpt-4o-mini", timeout=15.0)
        res = service.call_structured("sys", "user", SummaryResponse)
        self.assertEqual(res, expected_parsed)
        mock_client.beta.chat.completions.parse.assert_called_once()

    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_prompt_generation(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        service = LLMService()

        sys_s, user_s = service.get_summarization_prompt("test context")
        self.assertIn("topic-wise", user_s)

        sys_m, user_m = service.get_mcq_prompt("test context", num_questions=3)
        self.assertIn("Generate 3 MCQs", user_m)

        sys_p, user_p = service.get_planner_prompt(["Topic A", "Topic B"])
        self.assertIn("Topic A, Topic B", user_p)

if __name__ == '__main__':
    unittest.main()
