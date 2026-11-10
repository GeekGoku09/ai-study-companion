import os
import unittest

from src.utils import check_file_exists, count_tokens, format_tokens


class TestUtils(unittest.TestCase):
    def test_check_file_exists(self):
        self.assertTrue(check_file_exists(__file__))
        self.assertFalse(check_file_exists("non_existent_file_xyz.txt"))
        self.assertFalse(check_file_exists(os.path.dirname(__file__)))  # directory, not file

    def test_count_tokens(self):
        self.assertEqual(count_tokens(""), 0)
        tokens = count_tokens("Hello world! This is an AI test.")
        self.assertGreater(tokens, 0)
        self.assertEqual(format_tokens("Hello world!"), count_tokens("Hello world!"))

    def test_count_tokens_invalid_model_fallback(self):
        tokens = count_tokens("Fallback token counter test", model_or_encoding="invalid_model_xyz")
        self.assertGreater(tokens, 0)


if __name__ == "__main__":
    unittest.main()
