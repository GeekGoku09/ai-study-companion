import unittest
from unittest.mock import MagicMock, patch

from src.retriever import Retriever
from src.schemas import DocumentChunk


class TestRetriever(unittest.TestCase):
    def test_init_without_key_raises_error(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            Retriever(api_key=None)

    @patch("src.retriever.OpenAI")
    @patch("src.retriever.faiss.IndexFlatL2")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    def test_build_index(self, mock_faiss, mock_openai):
        mock_client = mock_openai.return_value

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

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    def test_retrieve_without_build_index_returns_empty(self):
        retriever = Retriever()
        result = retriever.retrieve("query", k=5)
        self.assertEqual(result, [])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    def test_build_index_empty_chunks(self):
        retriever = Retriever()
        retriever.build_index([])
        self.assertEqual(retriever.chunks, [])
        self.assertIsNone(retriever.index)

    @patch("src.retriever.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    def test_retrieve_with_data(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever()
        chunk1 = DocumentChunk(content="Calculus notes", metadata={"page": 1})
        chunk2 = DocumentChunk(content="Physics notes", metadata={"page": 2})
        retriever.build_index([chunk1, chunk2])

        results = retriever.retrieve("Calculus", k=1)
        self.assertEqual(len(results), 1)

        zero_results = retriever.retrieve("Calculus", k=0)
        self.assertEqual(len(zero_results), 0)


if __name__ == "__main__":
    unittest.main()
