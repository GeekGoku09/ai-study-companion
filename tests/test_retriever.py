import unittest
import tempfile
from pathlib import Path
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
    def test_retrieve_unbuilt_index_raises(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        with self.assertRaises(ValueError):
            retriever.retrieve("sample query")

    @patch('src.retriever.OpenAI')
    def test_save_and_load_index(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever(api_key="fake-key")
        chunks = [DocumentChunk(content="Calculus basics", metadata={"page": 1})]
        retriever.build_index(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "faiss_store"
            retriever.save_index(save_path)
            self.assertTrue((save_path / "index.faiss").exists())
            self.assertTrue((save_path / "chunks.json").exists())

            # Load into new retriever
            new_retriever = Retriever(api_key="fake-key")
            new_retriever.load_index(save_path)
            self.assertEqual(len(new_retriever.chunks), 1)
            self.assertEqual(new_retriever.chunks[0].content, "Calculus basics")
            self.assertEqual(new_retriever.dimension, 1536)

    @patch('src.retriever.OpenAI')
    def test_retrieve_empty_query(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        retriever.index = MagicMock()
        retriever.chunks = [DocumentChunk(content="Test", metadata={})]

        result = retriever.retrieve("")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
