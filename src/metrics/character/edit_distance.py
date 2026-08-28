"""Normalized edit distance character-level metric."""

from abc import ABC

import Levenshtein


class NormalizedEditDistance(ABC):
    """Normalized edit distance: 1 - (Levenshtein distance / max length)."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        edit_distance = Levenshtein.distance(prediction, reference)
        max_len = max(len(prediction), len(reference))
        return 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
