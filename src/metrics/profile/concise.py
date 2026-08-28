"""Concise profile metric."""

from utils.text import preprocess_text
from abc import ABC
import re
import math


class Concise(ABC):
    """
    Formula: 1 / (1 + log(max(words_per_sentence, 1)))
    Range: 0.0-1.0 (higher = more concise)
    Reference: Graesser et al. (2004). Behavior Research Methods, 36(2), 193-202.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        text_lower = text.lower()
        words, _ = preprocess_text(text_lower)
        word_count = len(words)
        sentences = len(re.split(r'[.!?]+', text))
        words_per_sentence = word_count / max(sentences, 1)
        score = 1 / (1 + math.log(max(words_per_sentence, 1)))
        return max(0.0, min(1.0, score))
