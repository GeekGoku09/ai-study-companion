import unittest
from unittest.mock import MagicMock, patch
from src.ingest import extract_text_from_pdf, semantic_chunking, ingest_pdf
from src.schemas import DocumentChunk

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

    @patch('os.path.exists')
    def test_extract_text_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("nonexistent.pdf")

    @patch('src.ingest.PdfReader')
    @patch('os.path.exists')
    def test_extract_text_encrypted_pdf(self, mock_exists, mock_pdf_reader):
        mock_exists.return_value = True
        mock_reader_instance = mock_pdf_reader.return_value
        mock_reader_instance.is_encrypted = True
        mock_reader_instance.decrypt.return_value = 0 # Decryption failed

        with self.assertRaises(ValueError) as ctx:
            extract_text_from_pdf("locked.pdf")
        self.assertIn("encrypted", str(ctx.exception).lower())

    def test_semantic_chunking(self):
        # Each chunk is ~30 tokens (less than max_tokens=100)
        chunks = [
            DocumentChunk(content="Paragraph 1 with concise technical content. " * 3, metadata={"page": 1}),
            DocumentChunk(content="Paragraph 2 with concise technical content. " * 3, metadata={"page": 1}),
            DocumentChunk(content="Paragraph 3 with concise technical content. " * 3, metadata={"page": 2}),
        ]
        
        # Max tokens 40 forces them to stay separate (each is ~30 tokens, 30+30 > 40)
        refined = semantic_chunking(chunks, max_tokens=40)
        self.assertEqual(len(refined), 3)
        
        # Max tokens large (1000) allows grouping on same page
        refined_large = semantic_chunking(chunks, max_tokens=1000)
        self.assertEqual(len(refined_large), 2)
        self.assertIn("Paragraph 1", refined_large[0].content)
        self.assertIn("Paragraph 2", refined_large[0].content)

    def test_semantic_chunking_oversized_paragraph(self):
        huge_text = "Detailed engineering concept explanation. " * 200
        oversized_chunk = DocumentChunk(content=huge_text, metadata={"page": 1, "chunk_id": "p1_c0"})

        refined = semantic_chunking([oversized_chunk], max_tokens=100)
        self.assertGreater(len(refined), 1)
        for chunk in refined:
            self.assertEqual(chunk.metadata["page"], 1)

    def test_empty_chunks(self):
        self.assertEqual(semantic_chunking([]), [])

if __name__ == '__main__':
    unittest.main()
