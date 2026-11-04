import unittest
from unittest.mock import MagicMock, patch
from src.main import main

class TestMainCLI(unittest.TestCase):
    def test_missing_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            exit_code = main(["--pdf", "sample.pdf", "--mode", "summary"])
            self.assertEqual(exit_code, 1)

    @patch('src.main.ingest_pdf')
    @patch('src.main.Retriever')
    @patch('src.main.LLMService')
    @patch('src.main.Summarizer')
    def test_main_summary_mode(self, mock_summarizer_cls, mock_llm_cls, mock_retriever_cls, mock_ingest):
        mock_ingest.return_value = [MagicMock(content="chunk 1")]
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = [MagicMock(content="chunk 1")]

        mock_sum_inst = mock_summarizer_cls.return_value
        mock_sum_inst.generate_summary.return_value = MagicMock(topics=[
            MagicMock(topic="Topic 1", summary="Summary 1")
        ])

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'}):
            exit_code = main(["--pdf", "test.pdf", "--mode", "summary"])
            self.assertEqual(exit_code, 0)
            mock_ingest.assert_called_once_with("test.pdf")
            mock_retriever.build_index.assert_called_once()
            mock_sum_inst.generate_summary.assert_called_once()

if __name__ == '__main__':
    unittest.main()
