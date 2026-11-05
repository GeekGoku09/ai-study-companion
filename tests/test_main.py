import unittest
from unittest.mock import MagicMock, patch
import sys
import tempfile
import os
from src.main import main
from src.schemas import SummaryResponse, TopicSummary, MCQList, MCQ, RevisionPlan, RevisionDay, DocumentChunk

class TestMainCLI(unittest.TestCase):
    @patch('src.main.ingest_pdf')
    @patch('src.main.Retriever')
    @patch('src.main.LLMService')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_main_summary_mode(self, mock_getenv, mock_exists, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_getenv.return_value = "fake-key"
        mock_exists.return_value = True

        mock_ingest.return_value = [DocumentChunk(content="Engineering text", metadata={"page": 1})]
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [DocumentChunk(content="Engineering text", metadata={"page": 1})]

        mock_llm = mock_llm_cls.return_value
        mock_llm.get_summarization_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = SummaryResponse(
            topics=[TopicSummary(topic="Mechanics", summary="Mechanics is the branch of physics.")]
        )

        with tempfile.NamedTemporaryFile(suffix=".md") as tmp_out:
            test_args = ["main.py", "--pdf", "sample.pdf", "--mode", "summary", "--output", tmp_out.name]
            with patch.object(sys, 'argv', test_args):
                main()

            with open(tmp_out.name, "r") as f:
                output_content = f.read()
            self.assertIn("Mechanics", output_content)

    @patch('src.main.ingest_pdf')
    @patch('src.main.Retriever')
    @patch('src.main.LLMService')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_main_mcq_mode(self, mock_getenv, mock_exists, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_getenv.return_value = "fake-key"
        mock_exists.return_value = True

        mock_ingest.return_value = [DocumentChunk(content="Engineering text", metadata={"page": 1})]
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [DocumentChunk(content="Engineering text", metadata={"page": 1})]

        mock_llm = mock_llm_cls.return_value
        mock_llm.get_mcq_prompt.return_value = ("sys", "user")
        mock_llm.call_structured.return_value = MCQList(questions=[
            MCQ(
                question="What is strain?",
                options=["Stress / E", "Force * Area", "Mass * Accel", "Work / Time"],
                correct_answer="Stress / E",
                explanation="Strain is proportional to stress via Young's modulus."
            )
        ])

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_out:
            test_args = ["main.py", "--pdf", "sample.pdf", "--mode", "mcq", "--num-mcq", "1", "--output", tmp_out.name, "--format", "json"]
            with patch.object(sys, 'argv', test_args):
                main()

            with open(tmp_out.name, "r") as f:
                output_content = f.read()
            self.assertIn("What is strain?", output_content)

    @patch('src.main.ingest_pdf')
    @patch('src.main.Retriever')
    @patch('src.main.LLMService')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_main_plan_mode(self, mock_getenv, mock_exists, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_getenv.return_value = "fake-key"
        mock_exists.return_value = True

        mock_ingest.return_value = [DocumentChunk(content="Engineering text", metadata={"page": 1})]
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [DocumentChunk(content="Engineering text", metadata={"page": 1})]

        mock_llm = mock_llm_cls.return_value
        mock_llm.get_summarization_prompt.return_value = ("sys summary", "user summary")
        mock_llm.get_planner_prompt.return_value = ("sys planner", "user planner")

        # First call is for summary topics, second is for plan
        mock_llm.call_structured.side_effect = [
            SummaryResponse(topics=[TopicSummary(topic="Robotics", summary="Robotics systems control.")]),
            RevisionPlan(plan=[RevisionDay(day=1, topics=["Robotics"], activities=["Kinematics"], estimated_time_hours=3.0)])
        ]

        test_args = ["main.py", "--pdf", "sample.pdf", "--mode", "plan"]
        with patch.object(sys, 'argv', test_args):
            main()

    @patch('os.getenv')
    def test_main_missing_key_exits(self, mock_getenv):
        mock_getenv.return_value = None
        test_args = ["main.py", "--pdf", "sample.pdf", "--mode", "summary"]
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)

if __name__ == '__main__':
    unittest.main()
