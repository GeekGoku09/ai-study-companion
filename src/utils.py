import os

def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath)

def format_tokens(text: str) -> int:
    """
    Placeholder for token counting if needed outside of tiktoken directly.
    In this implementation, we mostly use tiktoken in ingest.py.
    """
    return len(text.split()) # Rough estimate
