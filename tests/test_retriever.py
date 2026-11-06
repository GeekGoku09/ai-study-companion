import unittest
import tempfile
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
        mock_embedding.index = 0
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
    def test_build_and_retrieve(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value

        def fake_create(input, model):
            res = MagicMock()
            data_list = []
            for i, text in enumerate(input):
                item = MagicMock()
                item.index = i
                # Return slightly different dummy vectors
                item.embedding = [float(i + 1)] * 1536
                data_list.append(item)
            res.data = data_list
            return res

        mock_client.embeddings.create.side_effect = fake_create

        retriever = Retriever()
        chunks = [
            DocumentChunk(content="chunk 1", metadata={"page": 1}),
            DocumentChunk(content="chunk 2", metadata={"page": 2}),
        ]
        retriever.build_index(chunks)

        results = retriever.retrieve("query", k=1)
        self.assertEqual(len(results), 1)

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_retrieve_empty_or_uninitialized(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        retriever = Retriever()
        self.assertEqual(retriever.retrieve("query"), [])
        self.assertEqual(retriever.retrieve(""), [])

    @patch('src.retriever.OpenAI')
    @patch('os.getenv')
    def test_save_and_load_index(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "fake-key"
        mock_client = mock_openai.return_value
        mock_embedding = MagicMock()
        mock_embedding.index = 0
        mock_embedding.embedding = [0.1] * 128
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        retriever = Retriever()
        chunks = [DocumentChunk(content="persistent chunk", metadata={"page": 1})]
        retriever.build_index(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save_index(tmpdir)

            new_retriever = Retriever()
            new_retriever.load_index(tmpdir)
            self.assertEqual(len(new_retriever.chunks), 1)
            self.assertEqual(new_retriever.chunks[0].content, "persistent chunk")

if __name__ == '__main__':
    unittest.main()
