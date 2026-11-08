import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from src.llm import LLMService


class DummyModel(BaseModel):
    message: str


class TestLLMService(unittest.TestCase):
    def test_init_without_key_raises_error(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            LLMService(api_key=None)

    @patch("src.llm.OpenAI")
    def test_call_structured_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.parsed = DummyModel(message="Success")
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="test-key", model="gpt-4o-mini", timeout=30.0)
        result = service.call_structured("sys", "user", DummyModel)

        self.assertEqual(result.message, "Success")
        mock_client.beta.chat.completions.parse.assert_called_once()

    @patch("src.llm.OpenAI")
    def test_call_structured_parse_failure(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.parsed = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="test-key")
        with self.assertRaises(RuntimeError):
            service.call_structured("sys", "user", DummyModel)

    @patch("src.llm.OpenAI")
    def test_prompt_generators(self, _mock_openai):
        service = LLMService(api_key="test-key")
        sys_s, user_s = service.get_summarization_prompt("test context")
        self.assertIn("technical academic assistant", sys_s)
        self.assertIn("test context", user_s)

        _sys_m, user_m = service.get_mcq_prompt("mcq context", num_questions=3)
        self.assertIn("Generate 3 MCQs", user_m)

        _sys_p, user_p = service.get_planner_prompt(["Topic A", "Topic B"])
        self.assertIn("Topic A, Topic B", user_p)


if __name__ == "__main__":
    unittest.main()
