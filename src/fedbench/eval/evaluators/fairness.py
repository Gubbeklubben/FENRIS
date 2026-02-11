from typing import Dict
from .base import Evaluator
from ..context import EvalContext


class FairnessEvaluator(Evaluator):
    name = "fairness"

    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        # intentionally minimal placeholder
        if not ctx.sensitive_columns or ctx.target_column is None:
            return {}
        return {"enabled": 0.0}
