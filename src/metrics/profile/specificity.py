"""Specificity profile metric."""

from utils.text import get_stopwords, preprocess_text
from abc import ABC


class Specificity(ABC):
    """
    Formula: unique_content_words / content_words
    Range: 0.0-1.0 (higher = more specific)
    Reference: Li & Hovy (2014). EMNLP.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        words, _ = preprocess_text(text)
        stop_words = get_stopwords()
        content_words = [w for w in words if w not in stop_words]
        content_count = len(content_words)
        unique_content = len(set(content_words))
        return unique_content / max(content_count, 1)
