"""Exact match lexical similarity metric."""

from abc import ABC


class ExactMatch(ABC):
    """Exact match: 1.0 if prediction equals reference after stripping, else 0.0."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        return 1.0 if prediction.strip() == reference.strip() else 0.0
