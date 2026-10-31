import os
from typing import Optional
import tiktoken

_ENCODING_CACHE: dict = {}

def get_tokenizer(model_or_encoding: str = "cl100k_base") -> tiktoken.Encoding:
    """Returns a cached tiktoken Encoding instance."""
    if model_or_encoding not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model_or_encoding] = tiktoken.get_encoding(model_or_encoding)
        except Exception:
            try:
                _ENCODING_CACHE[model_or_encoding] = tiktoken.encoding_for_model(model_or_encoding)
            except Exception:
                _ENCODING_CACHE[model_or_encoding] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model_or_encoding]

def count_tokens(text: str, model_or_encoding: str = "cl100k_base") -> int:
    """Accurately counts tokens in a string using tiktoken."""
    if not text:
        return 0
    try:
        enc = get_tokenizer(model_or_encoding)
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic: approx 4 chars per token for English text
        return max(1, len(text) // 4)

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists and is a regular file."""
    return os.path.exists(filepath) and os.path.isfile(filepath)

def write_output_file(filepath: str, content: str) -> None:
    """Writes output string content to the destination file, creating directories if needed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def format_tokens(text: str) -> int:
    """Legacy helper maintained for backward compatibility."""
    return count_tokens(text)
