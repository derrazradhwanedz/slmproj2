"""Engagement profile metric."""

from utils.text import preprocess_text
from abc import ABC


class Engagement(ABC):
    """
    Formula: [(question_marks + interrogatives×0.5) / word_count] × 10
    Range: 0.0-1.0 (higher = more engaging)
    Reference: Graesser et al. (2003). Behavior Research Methods, 36(2), 193-202.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        text = text.lower()
        words, _ = preprocess_text(text)
        word_count = len(words)
        question_marks = text.count('?')
        interrogatives = sum(1 for w in words if w in ['what', 'how', 'why', 'when', 'where', 'who'])
        score = (question_marks + interrogatives * 0.5) / max(word_count, 1) * 10
        return min(1.0, score)
