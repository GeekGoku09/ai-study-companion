import os
import shutil
import tempfile
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
    def test_batching_and_save_load(self, mock_openai):
        mock_client = mock_openai.return_value

        def fake_create(input, model):
            res = MagicMock()
            res.data = [MagicMock(index=i, embedding=[0.05 * i] * 8) for i in range(len(input))]
            return res

        mock_client.embeddings.create.side_effect = fake_create

        retriever = Retriever(api_key="fake-key", batch_size=2)
        chunks = [
            DocumentChunk(content=f"Content chunk {i}", metadata={"page": i})
            for i in range(5)
        ]
        retriever.build_index(chunks)
        self.assertEqual(len(retriever.chunks), 5)
        # 5 items with batch_size=2 should call API 3 times (2, 2, 1)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)

        # Retrieve top 2
        results = retriever.retrieve("query", k=2)
        self.assertEqual(len(results), 2)

        # Save and load
        temp_dir = tempfile.mkdtemp()
        try:
            retriever.save_index(temp_dir)
            loaded_retriever = Retriever(api_key="fake-key")
            loaded_retriever.load_index(temp_dir)
            self.assertEqual(len(loaded_retriever.chunks), 5)
            self.assertEqual(loaded_retriever.chunks[0].content, "Content chunk 0")
        finally:
            shutil.rmtree(temp_dir)

    def test_retrieve_unbuilt_raises(self):
        retriever = Retriever(api_key="fake-key")
        with self.assertRaises(ValueError):
            retriever.retrieve("query")

    def test_save_empty_raises(self):
        retriever = Retriever(api_key="fake-key")
        with self.assertRaises(ValueError):
            retriever.save_index("dummy_path")

if __name__ == '__main__':
    unittest.main()
