"""Sentence-embedding semantic similarity metric."""

from abc import ABC

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSimilarity(ABC):
    """Cosine similarity between sentence embeddings of prediction and reference."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must both be str")
        pred_emb = self.model.encode([prediction])[0]
        ref_emb = self.model.encode([reference])[0]
        denom = np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb)
        if denom == 0:
            return 0.0
        return float(np.dot(pred_emb, ref_emb) / denom)
