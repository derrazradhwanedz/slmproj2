"""Abstract base interface for reasoning-mechanism prompt builders."""

from abc import ABC, abstractmethod


class BaseTemplate(ABC):
    """Base interface for constructing an LLM prompt from a question.

    Each reasoning mechanism (SCoT, MGCoT, ...) implements this interface
    so the evaluator can build prompts polymorphically without branching
    on mechanism type.
    """

    @abstractmethod
    def format(self, question: str) -> str:
        """Build the full prompt to send to the language model.

        Args:
            question: The raw question text.

        Returns:
            The formatted prompt string.
        """
        raise NotImplementedError
