"""Readability profile metric."""

from utils.text import preprocess_text
import re
from abc import ABC


class Readability(ABC):
    """
    Formula: Flesch-Kincaid = 206.835 - 1.015×(avg_word_length) - 84.6×(sentences/word_count)
    Range: 0-100 (higher = more readable)
    Reference: Flesch, R. (1948). Journal of Applied Psychology, 32(3), 221-233.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        text = text.lower()
        words, _ = preprocess_text(text)
        word_count = len(words)
        sentences = len(re.split(r'[.!?]+', text))
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        score = 206.835 - 1.015 * avg_word_length - 84.6 * (sentences / max(word_count, 1))
        return max(0.0, min(100.0, score))
