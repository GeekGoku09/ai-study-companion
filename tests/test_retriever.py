import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
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
    def test_retrieve_and_persistence(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # batch_size=2: 3 chunks -> 2 batches in build_index, then 1 call in retrieve
        mock_client.embeddings.create.side_effect = [
            # First batch (2 items)
            MagicMock(data=[
                MagicMock(embedding=[1.0, 0.0, 0.0]),
                MagicMock(embedding=[0.0, 1.0, 0.0]),
            ]),
            # Second batch (1 item)
            MagicMock(data=[
                MagicMock(embedding=[0.0, 0.0, 1.0]),
            ]),
            # Retrieve query embedding (1 item)
            MagicMock(data=[
                MagicMock(embedding=[1.0, 0.0, 0.0])
            ])
        ]

        retriever = Retriever(api_key="fake-key", batch_size=2)
        chunks = [
            DocumentChunk(content="Chunk 1 about mechanics", metadata={"page": 1}),
            DocumentChunk(content="Chunk 2 about circuits", metadata={"page": 2}),
            DocumentChunk(content="Chunk 3 about materials", metadata={"page": 3}),
        ]

        retriever.build_index(chunks)
        self.assertEqual(retriever.dimension, 3)
        self.assertEqual(len(retriever.chunks), 3)

        # Retrieve top 2
        results = retriever.retrieve("query mechanics", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].content, "Chunk 1 about mechanics")

        # Test save and load index
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save_index(tmpdir)

            loaded_retriever = Retriever(api_key="fake-key")
            loaded_retriever.load_index(tmpdir)
            self.assertEqual(len(loaded_retriever.chunks), 3)
            self.assertEqual(loaded_retriever.dimension, 3)
            self.assertEqual(loaded_retriever.chunks[0].content, "Chunk 1 about mechanics")

if __name__ == '__main__':
    unittest.main()
