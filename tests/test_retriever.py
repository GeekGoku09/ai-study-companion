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
    def test_retrieve_query_caching(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.05] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever()
        chunks = [
            DocumentChunk(content="Thermodynamics first law", metadata={"page": 1}),
            DocumentChunk(content="Thermodynamics second law", metadata={"page": 2}),
        ]
        retriever.build_index(chunks)

        # Initial call count after building index (1 call for batch chunk embeddings)
        initial_call_count = mock_client.embeddings.create.call_count
        self.assertEqual(initial_call_count, 1)

        # First query retrieve
        results1 = retriever.retrieve("What is the first law?", k=1)
        self.assertEqual(len(results1), 1)
        self.assertEqual(mock_client.embeddings.create.call_count, initial_call_count + 1)

        # Repeated identical query retrieve - should hit cache and NOT trigger OpenAI API call
        results2 = retriever.retrieve("What is the first law?", k=1)
        self.assertEqual(len(results2), 1)
        self.assertEqual(mock_client.embeddings.create.call_count, initial_call_count + 1)

        # Different query should call API
        results3 = retriever.retrieve("What is entropy?", k=1)
        self.assertEqual(len(results3), 1)
        self.assertEqual(mock_client.embeddings.create.call_count, initial_call_count + 2)

        # Re-building index should clear the query cache
        retriever.build_index(chunks)
        retriever.retrieve("What is the first law?", k=1)
        self.assertEqual(mock_client.embeddings.create.call_count, initial_call_count + 4)

if __name__ == '__main__':
    unittest.main()
