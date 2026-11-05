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
    def test_build_index_batching(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        # Prepare 5 chunks with batch_size=2 -> 3 calls
        def make_resp(num):
            resp = MagicMock()
            resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(num)]
            return resp

        mock_client.embeddings.create.side_effect = [make_resp(2), make_resp(2), make_resp(1)]

        retriever = Retriever(batch_size=2)
        chunks = [DocumentChunk(content=f"Content {i}", metadata={"page": i}) for i in range(5)]
        retriever.build_index(chunks)

        self.assertEqual(mock_client.embeddings.create.call_count, 3)
        self.assertEqual(len(retriever.chunks), 5)

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_retrieve_empty_or_no_index(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        res = retriever.retrieve("query")
        self.assertEqual(res, [])

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_retrieve_actual_index(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        # Mock embeddings
        emb1 = MagicMock(embedding=[1.0, 0.0])
        emb2 = MagicMock(embedding=[0.0, 1.0])
        mock_resp = MagicMock(data=[emb1, emb2])

        q_emb = MagicMock(embedding=[1.0, 0.0])
        mock_q_resp = MagicMock(data=[q_emb])

        mock_client.embeddings.create.side_effect = [mock_resp, mock_q_resp]

        retriever = Retriever()
        chunks = [
            DocumentChunk(content="Chunk A", metadata={"page": 1}),
            DocumentChunk(content="Chunk B", metadata={"page": 2})
        ]
        retriever.build_index(chunks)
        results = retriever.retrieve("Chunk A query", k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Chunk A")

if __name__ == '__main__':
    unittest.main()
