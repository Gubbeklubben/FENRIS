"""
Utility evaluators.

Measures the downstream predictive usefulness of the synthetic data
using a Train-on-Synthetic / Test-on-Real (TSTR) approach with
logistic regression (classification) or ridge regression (regression).
"""

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from fedbench.core.data import TableSchema
from fedbench.core.eval import Evaluator, LocalEvalContext
from fedbench.core.eval.evalcontext import GlobalEvalContext
from fedbench.util.metrics import fit_tabular_model, weighted_mean


class TSTREvaluator(Evaluator):
    # noinspection PyMethodMayBeStatic
    def _compute(
        self,
        D_test: pd.DataFrame,
        D_syn: pd.DataFrame,
        y_col: str | None,
        schema: TableSchema,
        seed: int,
    ) -> dict[str, float]:

        metrics = {
            "tstr_auc": math.nan,
            "tstr_accuracy": math.nan,
            "tstr_rmse": math.nan,
        }

        if y_col is None or D_test is None:
            return metrics

        X_syn = D_syn.drop(columns=[y_col])
        y_syn = D_syn[y_col]
        X_test = D_test.drop(columns=[y_col])
        y_test = D_test[y_col]

        if X_syn.empty:
            return metrics

        if schema.kind_of(y_col) in ["binary", "categorical"]:
            if y_syn.nunique() < 2:
                return metrics

            model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
            pipe = fit_tabular_model(X_syn, y_syn, model)

            if schema.kind_of(y_col) == "binary":
                y_proba = pipe.predict_proba(X_test)[:, 1]
                metrics["tstr_auc"] = roc_auc_score(y_test, y_proba)
            else:
                y_pred = pipe.predict(X_test)
                metrics["tstr_accuracy"] = accuracy_score(y_test, y_pred)

        else:
            model = Ridge(random_state=seed)
            pipe = fit_tabular_model(X_syn, y_syn, model)
            y_pred = pipe.predict(X_test)
            metrics["tstr_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))

        return metrics

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        return self._compute(
            D_test=ctx.holdout_df,
            D_syn=ctx.synthetic_df,
            y_col=ctx.target_column,
            schema=ctx.schema,
            seed=ctx.seed,
        )

    def local_evaluate(self, ctx: LocalEvalContext) -> tuple[dict[str, float], int]:
        return self._compute(
            D_test=ctx.test_df,
            D_syn=ctx.synthetic_df,
            y_col=ctx.target_column,
            schema=ctx.schema,
            seed=ctx.seed,
        ), len(ctx.test_df)

    def aggregate(
        self, stats: Iterable[tuple[Mapping[str, float], int]]
    ) -> dict[str, float]:
        keys = ("tstr_auc", "tstr_accuracy", "tstr_rmse")
        if not stats:
            return {key: math.nan for key in keys}

        pairs: dict[str, list[tuple[float, int]]] = {key: [] for key in keys}

        for metrics, n_test in stats:
            for key in keys:
                pairs[key].append((metrics[key], n_test))

        return {
            key: weighted_mean(pairs[key])  # nofmt
            for key in keys
        }
