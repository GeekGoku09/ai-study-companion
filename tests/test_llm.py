import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from openai import RateLimitError
from src.llm import LLMService

class DummyModel(BaseModel):
    message: str

class TestLLMService(unittest.TestCase):
    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_init_missing_key(self, mock_getenv, mock_openai):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            LLMService(api_key=None)

    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_call_structured_success(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        mock_parsed = DummyModel(message="hello")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(parsed=mock_parsed))]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService()
        result = service.call_structured("sys", "user", DummyModel)
        self.assertEqual(result.message, "hello")

    @patch('src.llm.OpenAI')
    @patch('os.getenv')
    def test_prompts(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        service = LLMService()

        sys_s, user_s = service.get_summarization_prompt("sample context")
        self.assertIn("engineering document", sys_s)
        self.assertIn("sample context", user_s)

        sys_m, user_m = service.get_mcq_prompt("sample context", 3)
        self.assertIn("3 MCQs", user_m)

        sys_p, user_p = service.get_planner_prompt(["Math", "Physics"])
        self.assertIn("Math, Physics", user_p)

if __name__ == "__main__":
    unittest.main()
