"""Entropy profile metric."""

from utils.text import preprocess_text
from abc import ABC
from collections import Counter
from scipy.stats import entropy
import numpy as np


class Entropy(ABC):
    """
    Formula: -sum(p_i x log2(p_i))
    Range: 0.0+ (higher = more uniform)
    Reference: Shannon (1948). Bell System Technical Journal.
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
        probs = np.array(list(Counter(words).values())) / word_count
        return entropy(probs)
