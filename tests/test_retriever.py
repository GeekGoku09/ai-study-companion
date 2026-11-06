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
    @patch('src.retriever.faiss.IndexFlatL2')
    @patch('os.getenv')
    def test_retrieve_and_query_cache(self, mock_getenv, mock_faiss, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        mock_index_instance = mock_faiss.return_value
        # Return index 0 as nearest neighbor
        mock_index_instance.search.return_value = ([[0.0]], [[0]])

        retriever = Retriever()
        chunk = DocumentChunk(content="Test content", metadata={"page": 1})
        retriever.build_index([chunk])

        # First retrieval should call OpenAI embedding API
        results1 = retriever.retrieve("What is test content?", k=1)
        self.assertEqual(len(results1), 1)
        self.assertEqual(results1[0].content, "Test content")
        self.assertEqual(mock_client.embeddings.create.call_count, 2)  # 1 for build_index, 1 for retrieve

        # Second retrieval with same query should hit cache and NOT call OpenAI API again
        results2 = retriever.retrieve("What is test content?", k=1)
        self.assertEqual(len(results2), 1)
        self.assertEqual(results2[0].content, "Test content")
        self.assertEqual(mock_client.embeddings.create.call_count, 2)  # Still 2, cached!

if __name__ == '__main__':
    unittest.main()
