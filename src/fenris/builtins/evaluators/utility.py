"""Utility evaluators.

Measures the downstream predictive usefulness of the synthetic data
using a Train-on-Synthetic / Test-on-Real (TSTR) approach with
logistic regression (classification) or ridge regression (regression).
"""

from typing import ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from fenris.builtins.evaluators._helpers import (
    fit_tabular_model,
    weighted_mean_metrics,
)
from fenris.core.data import TableSchema
from fenris.core.eval import Category, Evaluator, LocalEvalContext
from fenris.core.eval.evalcontext import GlobalEvalContext
from fenris.core.eval.evaluator import (
    EvaluationMode,
    EvaluatorSpec,
    MetricSpec,
)


class TSTREvaluator(Evaluator):
    EVALUATOR_SPEC: ClassVar[EvaluatorSpec] = EvaluatorSpec(
        category=Category.UTILITY,
        eval_mode=EvaluationMode.BOTH,
        metrics=[
            MetricSpec("tstr_auc", default_stop_mode="max"),
            MetricSpec("tstr_accuracy", default_stop_mode="max"),
            MetricSpec("tstr_rmse"),
        ],
    )

    # noinspection PyMethodMayBeStatic
    def _compute(
        self,
        d_test: pd.DataFrame,
        d_syn: pd.DataFrame,
        y_col: str | None,
        schema: TableSchema,
        seed: int,
    ) -> dict[str, float]:

        metrics = self._nan_result()

        if y_col is None or d_test is None:
            return metrics

        x_syn = d_syn.drop(columns=[y_col])
        y_syn = d_syn[y_col]
        x_test = d_test.drop(columns=[y_col])
        y_test = d_test[y_col]

        if x_syn.empty:
            return metrics

        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

        if schema.kind_of(y_col) in ["binary", "categorical"]:
            y_syn = y_syn.astype(str)
            y_test = y_test.astype(str)
            if y_syn.nunique() < 2:
                return metrics

            model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
            pipe = fit_tabular_model(x_syn, y_syn, model, schema)

            if schema.kind_of(y_col) == "binary":
                y_proba = pipe.predict_proba(x_test)[:, 1]
                metrics["tstr_auc"] = roc_auc_score(y_test, y_proba)
            else:
                y_pred = pipe.predict(x_test)
                metrics["tstr_accuracy"] = accuracy_score(y_test, y_pred)

        else:
            model = Ridge(random_state=seed)
            pipe = fit_tabular_model(x_syn, y_syn, model, schema)
            y_pred = pipe.predict(x_test)
            metrics["tstr_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))

        return metrics

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        return self._compute(
            d_test=ctx.holdout_df,
            d_syn=ctx.synthetic_df,
            y_col=ctx.target_column,
            schema=ctx.schema,
            seed=ctx.seed,
        )

    def local_evaluate(self, ctx: LocalEvalContext) -> tuple[dict[str, float], int]:
        return self._compute(
            d_test=ctx.test_df,
            d_syn=ctx.synthetic_df,
            y_col=ctx.target_column,
            schema=ctx.schema,
            seed=ctx.seed,
        ), len(ctx.test_df)

    def aggregate(
        self, stats: Iterable[tuple[Mapping[str, float], int]]
    ) -> dict[str, float]:
        return weighted_mean_metrics(stats, self.get_metric_keys())
