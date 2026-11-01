import os
from typing import List
import tiktoken

_DEFAULT_ENCODING = "cl100k_base"

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath)

def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """Accurately counts the number of tokens in a string using tiktoken."""
    if not text:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def format_tokens(text: str) -> int:
    """Deprecated: Backward-compatible token counting alias."""
    return count_tokens(text)

def split_text_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    encoding_name: str = _DEFAULT_ENCODING
) -> List[str]:
    """
    Splits a long text into chunks that do not exceed max_tokens.
    Useful for breaking down huge paragraphs that exceed chunk limits.
    """
    if not text:
        return []

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    step = max_tokens - overlap_tokens if max_tokens > overlap_tokens else max_tokens

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())

    return chunks
