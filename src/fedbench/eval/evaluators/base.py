from abc import ABC, abstractmethod
from typing import Dict

from fedbench.eval.context import EvalContext


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        pass