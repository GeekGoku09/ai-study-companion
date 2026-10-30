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
        mock_reader_instance.pages = [mock_page]
        
        chunks = extract_text_from_pdf("dummy.pdf")
        
        self.assertEqual(len(chunks), 2)
        self.assertIn("long test paragraph", chunks[0].content)
        self.assertIn("another long paragraph", chunks[1].content)

    def test_semantic_chunking(self):
        chunks = [
            DocumentChunk(content="Short paragraph 1.", metadata={"page": 1}),
            DocumentChunk(content="Short paragraph 2.", metadata={"page": 1}),
            DocumentChunk(content="Short paragraph 3.", metadata={"page": 2}),
        ]
        
        # Max tokens small enough to separate each short paragraph, but not sub-chunk them
        refined = semantic_chunking(chunks, max_tokens=6)
        
        # Each chunk is small enough, but combined they exceed 6 tokens, so they stay separate
        self.assertEqual(len(refined), 3)
        
        # Max tokens large to allow grouping
        refined_large = semantic_chunking(chunks, max_tokens=1000)
        # Para 1 and 2 should group (same page), Para 3 stays separate (new page)
        self.assertEqual(len(refined_large), 2)
        self.assertIn("Short paragraph 1.", refined_large[0].content)
        self.assertIn("Short paragraph 2.", refined_large[0].content)

    def test_semantic_chunking_oversized_paragraph(self):
        # Test sub-chunking of paragraphs larger than max_tokens
        chunks = [
            DocumentChunk(content="This is a long sentence with lots of words repeating. " * 30, metadata={"page": 1}),
        ]
        refined = semantic_chunking(chunks, max_tokens=50)
        self.assertGreater(len(refined), 1)
        for r in refined:
            self.assertEqual(r.metadata["page"], 1)
            self.assertIn("sub_chunk", r.metadata)

if __name__ == '__main__':
    unittest.main()
