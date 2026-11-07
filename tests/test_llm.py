import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from src.llm import LLMService


class DummyModel(BaseModel):
    message: str


class TestLLMService(unittest.TestCase):
    @patch("src.llm.OpenAI")
    @patch("os.getenv")
    def test_init_missing_key(self, mock_getenv, mock_openai):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError) as ctx:
            LLMService(api_key=None)
        self.assertIn("OpenAI API key is required", str(ctx.exception))

    @patch("src.llm.OpenAI")
    def test_init_custom_params(self, mock_openai):
        service = LLMService(api_key="sk-test", model="gpt-4o", timeout=45.0)
        self.assertEqual(service.model, "gpt-4o")
        self.assertEqual(service.timeout, 45.0)
        mock_openai.assert_called_once_with(api_key="sk-test", timeout=45.0)

    @patch("src.llm.OpenAI")
    def test_call_structured_success(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_choice = MagicMock()
        mock_choice.message.parsed = DummyModel(message="hello")
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="sk-test")
        result = service.call_structured("sys prompt", "user prompt", DummyModel)
        self.assertEqual(result.message, "hello")

    @patch("src.llm.OpenAI")
    def test_call_structured_parse_failure(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_choice = MagicMock()
        mock_choice.message.parsed = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="sk-test")
        with self.assertRaises(ValueError) as ctx:
            service.call_structured("sys prompt", "user prompt", DummyModel)
        self.assertIn("Failed to parse structured response", str(ctx.exception))

    @patch("src.llm.OpenAI")
    def test_prompt_generators(self, mock_openai):
        service = LLMService(api_key="sk-test")

        sys_s, user_s = service.get_summarization_prompt("context text")
        self.assertIn("engineering document chunks", sys_s)
        self.assertIn("context text", user_s)

        sys_m, user_m = service.get_mcq_prompt("formula context", num_questions=8)
        self.assertIn("examiner", sys_m)
        self.assertIn("8 MCQs", user_m)

        sys_p, user_p = service.get_planner_prompt(["Thermo", "Fluids"])
        self.assertIn("study coach", sys_p)
        self.assertIn("Thermo, Fluids", user_p)


if __name__ == "__main__":
    unittest.main()
