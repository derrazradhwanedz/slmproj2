"""Coherence profile metric."""

from utils.text import preprocess_text
import re
from abc import ABC


class Coherence(ABC):
    """
    Formula: 1 - (sentences / word_count)
    Range: 0.0-1.0 (higher = more coherent)
    Reference: Foltz et al. (1998). Discourse Processes, 25(2-3), 285-307.
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
        score = 1.0 - (sentences / max(word_count, 1))
        return max(0.0, min(1.0, score))
