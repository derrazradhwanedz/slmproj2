"""Perplexity profile metric."""

from metrics.profile.entropy import Entropy
from abc import ABC
import math


class Perplexity(ABC):
    """
    Formula: 2 ^ entropy
    Range: 1.0+ (lower = more predictable)
    Reference: Jelinek et al. (1977). IBM Research.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> float:
        entropy_calc = Entropy()
        entropy_value = entropy_calc(text)
        return math.pow(2, entropy_value) if entropy_value > 0 else 1.0
