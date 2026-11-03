import unittest
from unittest.mock import MagicMock, patch
from src.main import main

class TestMainCLI(unittest.TestCase):
    @patch('sys.argv', ['main.py', '--mode', 'summary'])
    @patch('src.main.os.getenv')
    def test_missing_pdf_and_load_index(self, mock_getenv):
        mock_getenv.return_value = "fake-key"
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)

    @patch('sys.argv', ['main.py', '--mode', 'summary', '--pdf', 'test.pdf'])
    @patch('src.main.os.getenv')
    def test_missing_openai_key(self, mock_getenv):
        mock_getenv.return_value = None
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)

    @patch('src.main.export_to_json')
    @patch('src.main.Summarizer')
    @patch('src.main.Retriever')
    @patch('src.main.ingest_pdf')
    @patch('src.main.LLMService')
    @patch('src.main.os.getenv')
    @patch('sys.argv', ['main.py', '--mode', 'summary', '--pdf', 'test.pdf', '--output', 'res.json'])
    def test_main_summary_flow(
        self,
        mock_getenv,
        mock_llm,
        mock_ingest,
        mock_retriever_cls,
        mock_summarizer_cls,
        mock_export_json
    ):
        mock_getenv.return_value = "fake-key"
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = []

        mock_summarizer = mock_summarizer_cls.return_value
        summary_result = MagicMock()
        summary_result.topics = []
        mock_summarizer.generate_summary.return_value = summary_result

        main()

        mock_ingest.assert_called_once_with('test.pdf')
        mock_retriever.build_index.assert_called_once()
        mock_export_json.assert_called_once()

if __name__ == "__main__":
    unittest.main()
