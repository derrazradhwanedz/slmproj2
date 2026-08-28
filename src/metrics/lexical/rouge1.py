"""ROUGE-1 F1 lexical similarity metric."""

from abc import ABC

from rouge_score import rouge_scorer


class Rouge1(ABC):
    """ROUGE-1 F1: unigram overlap between prediction and reference."""

    def __init__(self) -> None:
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        return self.scorer.score(prediction, reference)["rouge1"].fmeasure
