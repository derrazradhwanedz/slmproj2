"""Shared text preprocessing for profile metrics."""

from typing import List, Optional, Set, Tuple

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_stopwords_cache: Optional[Set[str]] = None


def preprocess_text(text: str) -> Tuple[List[str], List[str]]:
    """Common preprocessing for all profile metrics.

    Args:
        text: The input text.

    Returns:
        A tuple of (alphabetic words, all tokens), both lowercased.

    Raises:
        TypeError: If text is not a string.
        RuntimeError: If the NLTK 'punkt' data is missing.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    try:
        tokens = word_tokenize(text.lower())
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK 'punkt' data. Run nltk.download('punkt')."
        ) from exc

    words = [w for w in tokens if w.isalpha()]
    return words, tokens


def get_stopwords() -> Set[str]:
    """Return the cached English stopword set.

    Returns:
        The set of English stopwords.

    Raises:
        RuntimeError: If the NLTK 'stopwords' data is missing.
    """
    global _stopwords_cache
    if _stopwords_cache is None:
        try:
            _stopwords_cache = set(stopwords.words("english"))
        except LookupError as exc:
            raise RuntimeError(
                "Missing NLTK 'stopwords' data. Run nltk.download('stopwords')."
            ) from exc
    return _stopwords_cache
