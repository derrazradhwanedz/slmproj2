"""Relevance profile metric."""

from utils.text import get_stopwords, preprocess_text
from abc import ABC


class Relevance(ABC):
    """
    Formula: content_words / total_words
    Range: 0.0-1.0 (higher = more relevant)
    Reference: Mihalcea & Strapparava (2009). AAAI Proceedings.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        words, _ = preprocess_text(text)
        word_count = len(words)
        stop_words = get_stopwords()
        content_words = [w for w in words if w not in stop_words]
        return len(content_words) / max(word_count, 1)
