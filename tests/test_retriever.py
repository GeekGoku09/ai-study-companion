import unittest
from unittest.mock import MagicMock, patch
from src.retriever import Retriever
from src.schemas import DocumentChunk


class TestRetriever(unittest.TestCase):
    @patch('src.retriever.OpenAI')
    @patch('src.retriever.faiss.IndexFlatL2')
    @patch('os.getenv')
    def test_build_index(self, mock_getenv, mock_faiss, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value
        
        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.index = 0
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response
        
        retriever = Retriever()
        chunks = [DocumentChunk(content="Test content", metadata={"page": 1})]
        
        retriever.build_index(chunks)
        
        self.assertEqual(len(retriever.chunks), 1)
        mock_faiss.assert_called_once()

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_retrieve_without_index(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("test query")

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_retrieve_empty_query(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

    def test_missing_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                Retriever(api_key=None)


if __name__ == '__main__':
    unittest.main()
