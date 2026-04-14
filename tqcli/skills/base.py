"""Base skill class for builtin Python-implemented skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BuiltinSkill(ABC):
    """Base class for tqCLI built-in skills implemented in Python."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, args: list[str], context: dict) -> str:
        """Execute the skill and return output text."""
        ...

    @property
    def help_text(self) -> str:
        return self.description
