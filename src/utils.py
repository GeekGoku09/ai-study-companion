import logging
import os
import tiktoken
from typing import Optional

_ENCODING_CACHE = {}

def get_tokenizer(model_name: str = "cl100k_base"):
    """Returns a cached tiktoken tokenizer."""
    if model_name not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model_name] = tiktoken.get_encoding(model_name)
        except Exception:
            _ENCODING_CACHE[model_name] = tiktoken.encoding_for_model("gpt-4o-mini")
    return _ENCODING_CACHE[model_name]

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Accurately counts tokens in a string using tiktoken."""
    if not text:
        return 0
    encoding = get_tokenizer(encoding_name)
    return len(encoding.encode(text))

def setup_logger(name: str = "study_companion", level: Optional[str] = None) -> logging.Logger:
    """Configures and returns a structured logger."""
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        logger.setLevel(log_level)

    return logger

def sanitize_text(text: str) -> str:
    """Cleans up raw text: removes null characters, normalizes line endings and redundant whitespace."""
    if not text:
        return ""
    # Remove null and non-printable control characters except standard whitespace
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch == "\r" or ord(ch) >= 32)
    # Normalize carriage returns
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned.strip()
