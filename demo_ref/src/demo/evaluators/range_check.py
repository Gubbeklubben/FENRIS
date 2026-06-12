import math
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from fenris.core.eval import Category, Evaluator, LocalEvalContext
from fenris.core.eval.evalcontext import CentralizedEvalContext, GlobalEvalContext
from fenris.core.eval.evaluator import (
    EvaluationMode,
    EvaluatorSpec,
    MetricSpec,
)


class RangeCheck(Evaluator):
    EVALUATOR_SPEC: ClassVar[EvaluatorSpec] = EvaluatorSpec(
        category=Category.FIDELITY,
        eval_mode=EvaluationMode.CENTRALIZED,
        metrics=[MetricSpec("out_of_range_rate")],
    )

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        assert isinstance(ctx, CentralizedEvalContext)
        columns = ctx.schema.numeric_columns()

        stats = []
        for col in columns:
            r = ctx.client_train_df[col]
            s = ctx.synthetic_df[col]
            if r.empty or s.empty:
                continue
            out_of_range = (s < r.min()) | (s > r.max())
            stats.append(out_of_range.mean())

        return {"out_of_range_rate": float(np.mean(stats)) if stats else math.nan}

    def local_evaluate(self, ctx: LocalEvalContext) -> Any:
        return self._nan_result()

    def aggregate(self, stats: Iterable[Any]) -> dict[str, float]:
        return self._nan_result()
