"""Hapax profile metric."""

from utils.text import preprocess_text
from abc import ABC
from collections import Counter


class Hapax(ABC):
    """
    Formula: words_appearing_once / total_words
    Range: 0.0-1.0 (higher = more hapax legomena)
    Reference: Salton & Buckley (1988). Information Processing & Management.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        text_lower = text.lower()
        words, _ = preprocess_text(text_lower)
        word_count = len(words)
        if word_count == 0:
            return 0.0
        hapax_count = sum(1 for count in Counter(words).values() if count == 1)
        return hapax_count / word_count
