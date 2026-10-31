import os
import tempfile
import unittest
from src.utils import count_tokens, get_tokenizer, check_file_exists, write_output_file, format_tokens

class TestUtils(unittest.TestCase):
    def test_count_tokens(self):
        self.assertEqual(count_tokens(""), 0)
        tokens = count_tokens("Hello world! This is an AI study companion.")
        self.assertGreater(tokens, 0)
        self.assertEqual(format_tokens("Hello world"), count_tokens("Hello world"))

    def test_check_file_exists(self):
        self.assertTrue(check_file_exists("README.md"))
        self.assertFalse(check_file_exists("non_existent_file_xyz.txt"))
        self.assertFalse(check_file_exists("src")) # directory, not regular file

    def test_write_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "nested", "test_out.md")
            content = "# Sample Header\nTest content"
            write_output_file(out_path, content)

            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), content)

if __name__ == "__main__":
    unittest.main()
