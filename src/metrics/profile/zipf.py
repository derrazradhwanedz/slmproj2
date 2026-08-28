"""Zipf profile metric."""

from utils.text import preprocess_text
from abc import ABC
from collections import Counter
import numpy as np


class Zipf(ABC):
    """
    Formula: 1 / (1 + std(freq × rank))
    Range: 0.0-1.0 (higher = better Zipf compliance)
    Reference: Zipf (1949). Addison-Wesley.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        text_lower = text.lower()
        words, _ = preprocess_text(text_lower)
        word_freq = Counter(words)
        if len(word_freq) == 0:
            return 0.0
        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        zipf_scores = [freq * (rank + 1) for rank, (word, freq) in enumerate(sorted_freq)]
        std_score = np.std(zipf_scores)
        return max(0.0, min(1.0, 1 / (1 + std_score)))
