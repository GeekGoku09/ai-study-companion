import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from src.llm import LLMService


class DummyModel(BaseModel):
    message: str


class TestLLMService(unittest.TestCase):
    def test_missing_api_key_raises_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                LLMService(api_key=None)

    @patch("src.llm.OpenAI")
    def test_call_structured(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_choice = MagicMock()
        mock_choice.message.parsed = DummyModel(message="Success")
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.beta.chat.completions.parse.return_value = mock_response

        service = LLMService(api_key="fake-test-key", model="gpt-4o-mini")
        result = service.call_structured("sys prompt", "user prompt", DummyModel)

        self.assertIsInstance(result, DummyModel)
        self.assertEqual(result.message, "Success")
        mock_client.beta.chat.completions.parse.assert_called_once()

    def test_prompt_builders(self):
        service = LLMService(api_key="fake-test-key")
        sys_s, user_s = service.get_summarization_prompt("test context")
        self.assertIn("technical academic assistant", sys_s)
        self.assertIn("test context", user_s)

        sys_m, user_m = service.get_mcq_prompt("formula context", num_questions=3)
        self.assertIn("examiner", sys_m)
        self.assertIn("3 MCQs", user_m)

        sys_p, user_p = service.get_planner_prompt(["Math", "Physics"])
        self.assertIn("study coach", sys_p)
        self.assertIn("Math, Physics", user_p)


if __name__ == "__main__":
    unittest.main()
