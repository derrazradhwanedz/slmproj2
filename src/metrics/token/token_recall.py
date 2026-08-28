"""Token recall metric."""

from abc import ABC

from utils.text import preprocess_text


class TokenRecall(ABC):
    """Token recall: fraction of reference tokens that appear in the prediction."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        _, pred_tokens = preprocess_text(prediction)
        _, ref_tokens = preprocess_text(reference)
        pred_set, ref_set = set(pred_tokens), set(ref_tokens)
        common = len(pred_set & ref_set)
        return common / len(ref_set) if ref_set else 0.0
