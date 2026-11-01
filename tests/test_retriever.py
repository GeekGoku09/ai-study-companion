import unittest
import tempfile
import os
import numpy as np
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
    @patch('os.getenv')
    def test_embedding_batching(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        def fake_create(input, model):
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in input]
            return mock_resp

        mock_client.embeddings.create.side_effect = fake_create

        retriever = Retriever(batch_size=2)
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        embeddings = retriever._get_embeddings(texts)

        self.assertEqual(len(embeddings), 5)
        # With batch_size=2 and 5 items, create should be called 3 times
        self.assertEqual(mock_client.embeddings.create.call_count, 3)

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_retrieve_empty_or_no_index(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        # Empty query or unbuilt index returns empty list
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("query with no index"), [])

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_save_and_load_index(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever(dimension=4)
        chunks = [DocumentChunk(content="Thermodynamics overview", metadata={"page": 5})]
        retriever.build_index(chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            retriever.save_index(tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "index.faiss")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "metadata.json")))

            new_retriever = Retriever(dimension=4)
            new_retriever.load_index(tmp_dir)

            self.assertEqual(len(new_retriever.chunks), 1)
            self.assertEqual(new_retriever.chunks[0].content, "Thermodynamics overview")
            self.assertEqual(new_retriever.chunks[0].metadata["page"], 5)

if __name__ == '__main__':
    unittest.main()
