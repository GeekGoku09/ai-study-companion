import unittest
from src.utils import check_file_exists, count_tokens, format_tokens, split_text_by_tokens

class TestUtils(unittest.TestCase):
    def test_check_file_exists(self):
        self.assertTrue(check_file_exists("README.md"))
        self.assertFalse(check_file_exists("nonexistent_file_xyz.txt"))

    def test_count_tokens(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertGreater(count_tokens("Hello world! This is a test."), 0)
        # format_tokens backward compatibility
        self.assertEqual(format_tokens("Hello world!"), count_tokens("Hello world!"))

    def test_split_text_by_tokens(self):
        text = "Engineering systems requires deep architectural planning. " * 50
        total_tokens = count_tokens(text)
        self.assertGreater(total_tokens, 100)

        chunks = split_text_by_tokens(text, max_tokens=50)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk), 55) # Allow small margin for decoding boundary

    def test_split_text_empty(self):
        self.assertEqual(split_text_by_tokens(""), [])

if __name__ == "__main__":
    unittest.main()
