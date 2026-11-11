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
    @patch("os.getenv")
    def test_build_index_empty_chunks(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        retriever.build_index([])
        self.assertEqual(len(retriever.chunks), 0)
        self.assertIsNotNone(retriever.index)

    @patch("src.retriever.OpenAI")
    @patch("os.getenv")
    def test_retrieve_without_build_raises_error(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("sample query")

    @patch("src.retriever.OpenAI")
    @patch("os.getenv")
    def test_embedding_batching(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        def mock_create(input, model):
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.05] * 1536) for _ in input]
            return mock_resp

        mock_client.embeddings.create.side_effect = mock_create

        retriever = Retriever(batch_size=2)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        embeddings = retriever._get_embeddings(texts)

        self.assertEqual(len(embeddings), 5)
        # 5 items with batch_size=2 should take 3 calls (2, 2, 1)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)


if __name__ == "__main__":
    unittest.main()
