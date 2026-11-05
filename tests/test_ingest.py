import unittest
from unittest.mock import MagicMock, patch
from src.ingest import extract_text_from_pdf, semantic_chunking, ingest_pdf, IngestionError, EmptyDocumentError
from src.schemas import DocumentChunk
from pypdf.errors import PdfReadError

class TestIngest(unittest.TestCase):
    @patch('src.ingest.PdfReader')
    @patch('os.path.exists')
    def test_extract_text_from_pdf(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True
        
        # Mock PDF page
        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "This is a long test paragraph that should exceed the fifty character threshold for ingestion. " * 2 +
            "\n\n" +
            "This is another long paragraph that also exceeds the threshold to ensure it is not skipped." * 2
        )
        
        mock_reader_instance = mock_pdf_reader.return_value
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [mock_page]
        
        chunks = extract_text_from_pdf("dummy.pdf")
        
        self.assertEqual(len(chunks), 2)
        self.assertIn("long test paragraph", chunks[0].content)
        self.assertIn("another long paragraph", chunks[1].content)

    @patch('src.ingest.PdfReader')
    @patch('os.path.exists')
    def test_extract_text_empty_document_raises(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True
        mock_reader_instance = mock_pdf_reader.return_value
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = []

        with self.assertRaises(EmptyDocumentError):
            extract_text_from_pdf("empty.pdf")

    @patch('src.ingest.PdfReader')
    @patch('os.path.exists')
    def test_extract_text_corrupt_pdf_raises(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True
        mock_pdf_reader.side_effect = PdfReadError("Corrupt stream")

        with self.assertRaises(IngestionError):
            extract_text_from_pdf("corrupt.pdf")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/dummy.pdf")

    def test_semantic_chunking(self):
        chunks = [
            DocumentChunk(content="Para 1" * 50, metadata={"page": 1}), # approx 100-200 tokens
            DocumentChunk(content="Para 2" * 50, metadata={"page": 1}),
            DocumentChunk(content="Para 3" * 50, metadata={"page": 2}),
        ]
        
        # Max tokens very small to force split
        refined = semantic_chunking(chunks, max_tokens=50)
        
        # Each original chunk is > 50 tokens, so they should stay separate
        self.assertEqual(len(refined), 3)
        
        # Max tokens large to allow grouping
        refined_large = semantic_chunking(chunks, max_tokens=1000)
        # Para 1 and 2 should group (same page), Para 3 stays separate (new page)
        self.assertEqual(len(refined_large), 2)
        self.assertIn("Para 1", refined_large[0].content)
        self.assertIn("Para 2", refined_large[0].content)

    def test_semantic_chunking_oversized_paragraph(self):
        # A very large single paragraph
        long_content = "This is an important engineering concept with formulas and details. " * 80
        chunks = [DocumentChunk(content=long_content, metadata={"page": 1})]

        refined = semantic_chunking(chunks, max_tokens=100, split_oversized=True)
        self.assertGreater(len(refined), 1)

if __name__ == '__main__':
    unittest.main()
