"""BERTScore F1 semantic similarity metric."""

from abc import ABC

from bert_score import score


class BertScoreF1(ABC):
    """BERTScore F1: contextual embedding similarity between prediction and reference."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        if not prediction.strip() or not reference.strip():
            return 0.0
        try:
            _, _, f1 = score([prediction], [reference], lang="en", verbose=False)
            return float(f1.item())
        except Exception:
            return 0.0
