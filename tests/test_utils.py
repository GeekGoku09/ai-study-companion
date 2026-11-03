import unittest
import tempfile
import json
from pathlib import Path
from src.utils import (
    count_tokens,
    format_tokens,
    check_file_exists,
    validate_pdf_path,
    export_to_json,
    export_to_markdown,
)
from src.schemas import DocumentChunk

class TestUtils(unittest.TestCase):
    def test_token_counting(self):
        text = "Hello world, this is an AI Study Companion test."
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertEqual(count, format_tokens(text))
        self.assertEqual(count_tokens(""), 0)

    def test_check_file_exists(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.assertTrue(check_file_exists(tmp.name))
        self.assertFalse(check_file_exists("/path/to/nonexistent/file.pdf"))

    def test_validate_pdf_path(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with self.assertRaises(ValueError):
                validate_pdf_path(tmp.name)

        with self.assertRaises(FileNotFoundError):
            validate_pdf_path("missing_file.pdf")

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            # 0 byte file
            with self.assertRaises(ValueError):
                validate_pdf_path(tmp.name)

            tmp.write(b"%PDF-1.4 test content")
            tmp.flush()
            valid = validate_pdf_path(tmp.name)
            self.assertEqual(valid, Path(tmp.name))

    def test_export_to_json(self):
        chunk = DocumentChunk(content="Sample content", metadata={"page": 1})
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "output.json"
            export_to_json(chunk, out_file)
            self.assertTrue(out_file.exists())
            with open(out_file, "r") as f:
                data = json.load(f)
            self.assertEqual(data["content"], "Sample content")

    def test_export_to_markdown(self):
        content = "# Revision Summary\n\n- Key Point 1"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "output.md"
            export_to_markdown(content, out_file)
            self.assertTrue(out_file.exists())
            with open(out_file, "r") as f:
                text = f.read()
            self.assertEqual(text, content)

if __name__ == "__main__":
    unittest.main()
