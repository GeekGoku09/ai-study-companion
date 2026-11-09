import unittest
from unittest.mock import MagicMock
from src.mcq_generator import MCQGenerator
from src.llm import LLMService
from src.schemas import MCQList, MCQ, DocumentChunk


class TestMCQGenerator(unittest.TestCase):
    def test_generate_mcqs_schema(self):
        # Mock LLM Service
        mock_llm = MagicMock(spec=LLMService)
        
        expected_mcqs = MCQList(questions=[
            MCQ(
                question="What is 2+2?",
                options=["1", "2", "3", "4"],
                correct_answer="4",
                explanation="Basic addition."
            )
        ])
        
        mock_llm.get_mcq_prompt.return_value = ("system", "user")
        mock_llm.call_structured.return_value = expected_mcqs
        
        generator = MCQGenerator(mock_llm)
        chunks = [DocumentChunk(content="Context about math.", metadata={})]
        
        result = generator.generate_mcqs(chunks, num_questions=1)
        
        self.assertIsInstance(result, MCQList)
        self.assertEqual(len(result.questions), 1)
        self.assertEqual(result.questions[0].correct_answer, "4")

    def test_generate_mcqs_empty_chunks(self):
        mock_llm = MagicMock(spec=LLMService)
        generator = MCQGenerator(mock_llm)
        result = generator.generate_mcqs([])
        self.assertEqual(len(result.questions), 0)
        mock_llm.call_structured.assert_not_called()


if __name__ == '__main__':
    unittest.main()
