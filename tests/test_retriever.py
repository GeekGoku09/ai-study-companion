import unittest
from unittest.mock import MagicMock, patch

from src.retriever import Retriever
from src.schemas import DocumentChunk


class TestRetriever(unittest.TestCase):
    @patch("src.retriever.OpenAI")
    @patch("src.retriever.faiss.IndexFlatL2")
    @patch("os.getenv")
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

    @patch("src.retriever.OpenAI")
    def test_build_index_empty(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        retriever.build_index([])
        self.assertIsNone(retriever.index)
        self.assertEqual(retriever.chunks, [])

    @patch("src.retriever.OpenAI")
    def test_retrieve_without_build_raises(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        with self.assertRaises(ValueError):
            retriever.retrieve("sample query")

    @patch("src.retriever.OpenAI")
    def test_retrieve_empty_query(self, mock_openai):
        retriever = Retriever(api_key="fake-key")
        self.assertEqual(retriever.retrieve("   "), [])

    @patch("src.retriever.OpenAI")
    def test_batching_embeddings(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(5)]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever(api_key="fake-key")
        # Generate 15 items with batch_size=5 -> 3 calls
        texts = [f"Text {i}" for i in range(15)]
        embeddings = retriever._get_embeddings(texts, batch_size=5)

        self.assertEqual(mock_client.embeddings.create.call_count, 3)
        self.assertEqual(len(embeddings), 15)


if __name__ == "__main__":
    unittest.main()
