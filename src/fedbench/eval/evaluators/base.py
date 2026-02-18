import re
from abc import ABC, abstractmethod

from fedbench.eval.context import EvalContext


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> float:
        ...