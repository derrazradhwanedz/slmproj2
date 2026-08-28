"""Token F1 metric."""

from abc import ABC

from utils.text import preprocess_text


class TokenF1(ABC):
    """Token F1: harmonic mean of token precision and recall."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        _, pred_tokens = preprocess_text(prediction)
        _, ref_tokens = preprocess_text(reference)
        pred_set, ref_set = set(pred_tokens), set(ref_tokens)
        common = len(pred_set & ref_set)
        precision = common / len(pred_set) if pred_set else 0.0
        recall = common / len(ref_set) if ref_set else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
