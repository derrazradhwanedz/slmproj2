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
        _, _, f1 = score([prediction], [reference], lang="en", verbose=False)
        return f1.item()
