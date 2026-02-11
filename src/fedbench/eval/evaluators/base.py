from abc import ABC, abstractmethod
from typing import Dict

from ..context import EvalContext


class Evaluator(ABC):
    """Base class for all evaluators."""

    name: str  # e.g. "fidelity", "utility"

    def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        raw = self._evaluate(ctx)
        out: Dict[str, float] = {}

        for k, v in raw.items():
            out[f"{self.name}.{k}"] = float(v) if v is not None else None

        return out

    @abstractmethod
    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        ...
