"""Character F1 metric."""

from abc import ABC


class CharF1(ABC):
    """Character F1: harmonic mean of character-set precision and recall."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        char_pred = set(prediction.lower())
        char_ref = set(reference.lower())
        char_common = len(char_pred & char_ref)
        char_precision = char_common / len(char_pred) if char_pred else 0.0
        char_recall = char_common / len(char_ref) if char_ref else 0.0
        if (char_precision + char_recall) == 0:
            return 0.0
        return 2 * char_precision * char_recall / (char_precision + char_recall)
