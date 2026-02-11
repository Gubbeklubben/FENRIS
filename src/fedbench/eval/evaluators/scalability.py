from typing import Dict
from .base import Evaluator
from ..context import EvalContext


class ScalabilityEvaluator(Evaluator):
    name = "scalability"

    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        # runner injects wall-clock timing
        return {}
