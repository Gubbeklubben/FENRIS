import re
from abc import ABC, abstractmethod
from typing import Dict

from ..context import EvalContext


class Evaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    @property
    def category(self):
        ...

    @property
    def name(self):
        # Get class name
        cls_name = self.__class__.__name__

        # Remove 'Evaluator' suffix
        if not cls_name.endswith("Evaluator"):
            raise ValueError("Evaluator implementations must have a name ending in 'Evaluator'.")
        cls_name = cls_name[:-len("Evaluator")]

        # Convert CamelCase to snake_case
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', cls_name).lower()
        return snake

    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> float:
        ...