"""Token overlap count metric."""

from abc import ABC

from utils.text import preprocess_text


class TokenOverlapCount(ABC):
    """Token overlap count: number of distinct tokens shared between prediction and reference."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> int:
        _, pred_tokens = preprocess_text(prediction)
        _, ref_tokens = preprocess_text(reference)
        return len(set(pred_tokens) & set(ref_tokens))
