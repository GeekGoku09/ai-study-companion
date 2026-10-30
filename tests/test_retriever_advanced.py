import unittest
import tempfile
import os
import shutil
from unittest.mock import MagicMock, patch
from src.retriever import Retriever
from src.schemas import DocumentChunk
from src.exceptions import VectorIndexError

class TestRetrieverAdvanced(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.retriever.OpenAI')
    def test_save_and_load_index(self, mock_openai):
        mock_client = mock_openai.return_value
        # Mock embeddings
        emb1 = MagicMock()
        emb1.embedding = [0.1] * 128
        emb2 = MagicMock()
        emb2.embedding = [0.2] * 128
        mock_response = MagicMock()
        mock_response.data = [emb1, emb2]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever(api_key="fake-key")
        retriever.dimension = 128
        chunks = [
            DocumentChunk(content="First test chunk", metadata={"page": 1}),
            DocumentChunk(content="Second test chunk", metadata={"page": 2})
        ]
        retriever.build_index(chunks)
        self.assertEqual(retriever.index.ntotal, 2)

        # Save index
        save_path = os.path.join(self.temp_dir, "test_idx")
        retriever.save_index(save_path)
        self.assertTrue(os.path.exists(os.path.join(save_path, "index.faiss")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "chunks.json")))

        # Load index in a new instance
        retriever2 = Retriever(api_key="fake-key")
        retriever2.load_index(save_path)
        self.assertEqual(len(retriever2.chunks), 2)
        self.assertEqual(retriever2.index.ntotal, 2)
        self.assertEqual(retriever2.chunks[0].content, "First test chunk")

    @patch('src.retriever.OpenAI')
    def test_retrieve_empty_query(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        result = retriever.retrieve("")
        self.assertEqual(result, [])

    @patch('src.retriever.OpenAI')
    def test_retrieve_without_index_raises(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        with self.assertRaises(VectorIndexError):
            retriever.retrieve("some query")

if __name__ == "__main__":
    unittest.main()
