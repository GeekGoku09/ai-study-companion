import unittest
from unittest.mock import MagicMock, patch

from pypdf.errors import PdfReadError

from src.ingest import extract_text_from_pdf, ingest_pdf, semantic_chunking
from src.schemas import DocumentChunk


class TestIngest(unittest.TestCase):
    def test_extract_nonexistent_pdf(self):
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("nonexistent_test_file.pdf")

    @patch("src.ingest.PdfReader")
    @patch("os.path.exists")
    def test_extract_corrupted_pdf_raises_value_error(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True
        mock_pdf_reader.side_effect = PdfReadError("Corrupted stream")

        with self.assertRaises(ValueError) as ctx:
            extract_text_from_pdf("corrupt.pdf")
        self.assertIn("Failed to read PDF file", str(ctx.exception))

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
        mock_reader_instance.pages = [mock_page]

        chunks = extract_text_from_pdf("dummy.pdf")

        self.assertEqual(len(chunks), 2)
        self.assertIn("long test paragraph", chunks[0].content)
        self.assertIn("another long paragraph", chunks[1].content)

    def test_semantic_chunking_empty(self):
        self.assertEqual(semantic_chunking([]), [])

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

    @patch("src.ingest.extract_text_from_pdf")
    def test_ingest_pdf(self, mock_extract):
        mock_extract.return_value = [
            DocumentChunk(content="A" * 60, metadata={"page": 1})
        ]
        res = ingest_pdf("dummy.pdf")
        self.assertEqual(len(res), 1)


if __name__ == "__main__":
    unittest.main()
