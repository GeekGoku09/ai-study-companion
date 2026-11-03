import os
import json
from pathlib import Path
from typing import Any, Union
from pydantic import BaseModel
import tiktoken

_DEFAULT_ENCODING = None

def get_token_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Returns a cached tiktoken encoding instance."""
    global _DEFAULT_ENCODING
    if _DEFAULT_ENCODING is None:
        _DEFAULT_ENCODING = tiktoken.get_encoding(encoding_name)
    return _DEFAULT_ENCODING

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Accurately counts the number of tokens in text using tiktoken."""
    if not text:
        return 0
    enc = get_token_encoding(encoding_name)
    return len(enc.encode(text))

def format_tokens(text: str) -> int:
    """Count tokens in text using the standard tokenizer."""
    return count_tokens(text)

def check_file_exists(filepath: Union[str, Path]) -> bool:
    """Checks if a file exists and is a regular readable file."""
    return os.path.exists(str(filepath))

def validate_pdf_path(filepath: Union[str, Path]) -> Path:
    """Validates that the provided path exists and has a .pdf extension."""
    if not os.path.exists(str(filepath)):
        raise FileNotFoundError(f"File not found: {filepath}")

    p = Path(filepath)
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"Target file must be a PDF, got: {p.suffix}")

    try:
        if os.path.isfile(str(filepath)) and os.path.getsize(str(filepath)) == 0:
            raise ValueError(f"PDF file is empty (0 bytes): {filepath}")
    except OSError:
        pass

    return p

def export_to_json(model_or_data: Any, output_path: Union[str, Path]) -> None:
    """Exports a Pydantic model or dict to a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_or_data, BaseModel):
        data = model_or_data.model_dump()
    elif isinstance(model_or_data, (dict, list)):
        data = model_or_data
    else:
        raise TypeError(f"Unsupported data type for JSON export: {type(model_or_data)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def export_to_markdown(content: str, output_path: Union[str, Path]) -> None:
    """Exports raw markdown content to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
