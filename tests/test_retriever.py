import unittest
from unittest.mock import MagicMock, patch

from src.retriever import Retriever
from src.schemas import DocumentChunk


class TestRetriever(unittest.TestCase):
    @patch("src.retriever.OpenAI")
    @patch("src.retriever.faiss.IndexFlatIP")
    @patch("src.retriever.faiss.normalize_L2")
    @patch("os.getenv")
    def test_build_index(self, mock_getenv, mock_normalize, mock_faiss, mock_openai):
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
        mock_normalize.assert_called_once()

    @patch("src.retriever.OpenAI")
    def test_build_index_empty(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        retriever.build_index([])
        self.assertEqual(retriever.chunks, [])
        self.assertIsNone(retriever.index)

    @patch("src.retriever.OpenAI")
    def test_batching_embeddings(self, mock_openai):
        mock_client = mock_openai.return_value

        def fake_create(input, model):
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.05] * 1536) for _ in input]
            return mock_resp

        mock_client.embeddings.create.side_effect = fake_create

        retriever = Retriever(api_key="fake-key", batch_size=2)
        chunks = [DocumentChunk(content=f"Content {i}") for i in range(5)]

        retriever.build_index(chunks)
        # 5 items with batch size 2 -> 3 calls (2 + 2 + 1)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)
        self.assertEqual(len(retriever.chunks), 5)

    @patch("src.retriever.OpenAI")
    def test_retrieve_validation(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        with self.assertRaises(ValueError):
            retriever.retrieve("query")

        chunks = [DocumentChunk(content="Data")]
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai.return_value.embeddings.create.return_value = mock_resp
        retriever.build_index(chunks)

        with self.assertRaises(ValueError):
            retriever.retrieve("query", k=0)


if __name__ == "__main__":
    unittest.main()
