"""BLEU lexical similarity metric."""

from abc import ABC

from nltk.translate.bleu_score import sentence_bleu


class Bleu(ABC):
    """BLEU: n-gram precision with brevity penalty."""

    def __init__(self) -> None:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        reference_tokens = [reference.split()]
        prediction_tokens = prediction.split()
        return sentence_bleu(reference_tokens, prediction_tokens) # type: ignore
