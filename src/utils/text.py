"""Shared text preprocessing for profile metrics."""

from typing import List, Optional, Set, Tuple

import nltk
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
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(text.lower())

    words = [w for w in tokens if w.isalpha()]
    return words, tokens


def get_stopwords() -> Set[str]:
    """Return the cached English stopword set.

    Returns:
        The set of English stopwords.
    """
    global _stopwords_cache
    if _stopwords_cache is None:
        try:
            _stopwords_cache = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords_cache = set(stopwords.words("english"))
    return _stopwords_cache
