"""Template for implementing a new reasoning-mechanism prompt builder.

Copy this file, rename the class, and fill in the system instruction to add
a new mechanism (e.g. app/prompts/scot.py, app/prompts/mgcot.py).
"""
import textwrap
from typing import Literal
from base import BaseTemplate


class Template(BaseTemplate):
    """Prompt builder skeleton for a single reasoning mechanism."""

    def __init__(self):
        """Initialize the prompt builder with empty system/user buffers."""
        self.system: str = ""
        self.user: str = ""
        self.template: str = """
<|system|>
{system_instruction}

<|user|>
{question}

<|assistant|>

"""

    def add_text(self, text: str, mode: Literal["system", "user"], ident: int = 0) -> str:
        """Append text to the system or user buffer.

        Args:
            text: The text to append.
            mode: Which buffer to append to, "system" or "user".
            ident: Number of 4-space indentation levels to apply to every
                line of text before appending.

        Returns:
            The updated buffer content.

        Raises:
            ValueError: If mode is neither "system" nor "user".
        """
        if ident:
            text = textwrap.indent(text, " " * 4 * ident)
        if mode == "system":
            self.system += text
            return self.system
        if mode == "user":
            self.user += text
            return self.user
        raise ValueError(f"Invalid mode: {mode!r}. Expected 'system' or 'user'.")

    def format(self, question: str) -> str:
        """Build the full prompt to send to the language model.

        Args:
            question: The raw question text.

        Returns:
            The formatted prompt string.
        """
        self.add_text(question, mode="user")
        return self.template.format(
            system_instruction=self.system,
            question=self.user,
        )
