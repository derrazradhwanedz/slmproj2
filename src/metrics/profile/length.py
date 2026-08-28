"""Length profile metric."""

from utils.text import preprocess_text
from abc import ABC
import math


class Length(ABC):
    """
    Formula: 1 - e^(-0.05 × normalized_word_count)
    Range: 0.0-1.0 (peaks ~100 words)
    Reference: Burstein et al. (2013). Routledge.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        text_lower = text.lower()
        words, _ = preprocess_text(text_lower)
        word_count = len(words)
        normalized_count = word_count / 100
        k = 0.05
        score = 1 - math.exp(-k * normalized_count)
        return max(0.0, min(1.0, score))
