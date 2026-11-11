import unittest
from unittest.mock import MagicMock, patch
from pypdf.errors import PdfReadError
from src.ingest import extract_text_from_pdf, ingest_pdf, semantic_chunking
from src.schemas import DocumentChunk


class TestIngest(unittest.TestCase):
    @patch("src.ingest.PdfReader")
    @patch("os.path.exists")
    def test_extract_text_from_pdf(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True

        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "This is a long test paragraph that should exceed the fifty character threshold for ingestion. " * 2
            + "\n\n"
            + "This is another long paragraph that also exceeds the threshold to ensure it is not skipped." * 2
        )

        mock_reader_instance = mock_pdf_reader.return_value
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [mock_page]

        chunks = extract_text_from_pdf("dummy.pdf")

        self.assertEqual(len(chunks), 2)
        self.assertIn("long test paragraph", chunks[0].content)
        self.assertIn("another long paragraph", chunks[1].content)

    def test_extract_text_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("definitely_non_existent.pdf")

    @patch("src.ingest.PdfReader")
    @patch("os.path.exists")
    def test_extract_text_encrypted_failure(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True
        mock_reader = mock_pdf_reader.return_value
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = 0  # Failed decrypt

        with self.assertRaises(PdfReadError):
            extract_text_from_pdf("encrypted.pdf")

    def test_semantic_chunking(self):
        chunks = [
            DocumentChunk(content="Para 1" * 50, metadata={"page": 1}),
            DocumentChunk(content="Para 2" * 50, metadata={"page": 1}),
            DocumentChunk(content="Para 3" * 50, metadata={"page": 2}),
        ]

        refined = semantic_chunking(chunks, max_tokens=50)
        self.assertEqual(len(refined), 3)

        refined_large = semantic_chunking(chunks, max_tokens=1000)
        self.assertEqual(len(refined_large), 2)
        self.assertIn("Para 1", refined_large[0].content)
        self.assertIn("Para 2", refined_large[0].content)

    def test_semantic_chunking_empty(self):
        self.assertEqual(semantic_chunking([]), [])

    @patch("src.ingest.extract_text_from_pdf")
    def test_ingest_pdf_empty(self, mock_extract):
        mock_extract.return_value = []
        result = ingest_pdf("empty.pdf")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
