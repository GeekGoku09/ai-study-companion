import logging
import os

import tiktoken

logger = logging.getLogger(__name__)


def check_file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath) and os.path.isfile(filepath)


def count_tokens(text: str, model_or_encoding: str = "cl100k_base") -> int:
    """Accurately counts tokens for a text using tiktoken with fallback."""
    if not text:
        return 0
    try:
        if model_or_encoding.startswith(("gpt-", "text-")):
            encoding = tiktoken.encoding_for_model(model_or_encoding)
        else:
            encoding = tiktoken.get_encoding(model_or_encoding)
        return len(encoding.encode(text))
    except (KeyError, ValueError, TypeError) as e:
        logger.debug("Failed to encode using %s: %s; falling back to cl100k_base", model_or_encoding, e)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except (KeyError, ValueError, TypeError) as fallback_e:
            logger.debug("cl100k_base encoding fallback failed: %s; falling back to split", fallback_e)
            return max(1, len(text.split()))


def format_tokens(text: str) -> int:
    """Token count helper maintained for backward compatibility."""
    return count_tokens(text)
