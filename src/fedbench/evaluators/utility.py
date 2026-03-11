"""
Utility evaluators.

Measures the downstream predictive usefulness of the synthetic data
using a Train-on-Synthetic / Test-on-Real (TSTR) approach with
logistic regression (classification) or ridge regression (regression).
"""

import math
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from fedbench.core.eval import Evaluator, GlobalEvalContext, LocalEvalContext
from fedbench.util.metrics import fit_tabular_model


class TSTREvaluator(Evaluator):
    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        metrics = {
            "tstr_auc": math.nan,
            "tstr_accuracy": math.nan,
            "tstr_rmse": math.nan,
        }

        if ctx.target_column is None or ctx.holdout_df is None:
            return metrics

        D_syn = ctx.synthetic_df
        D_test = ctx.holdout_df
        y_col = ctx.target_column

        X_syn = D_syn.drop(columns=[y_col])
        y_syn = D_syn[y_col]
        X_test = D_test.drop(columns=[y_col])
        y_test = D_test[y_col]

        if X_syn.empty:
            return metrics

        if ctx.schema.kind_of(y_col) in ["binary", "categorical"]:
            if y_syn.nunique() < 2:
                return metrics

            model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=ctx.seed,
            )
            pipe = fit_tabular_model(X_syn, y_syn, model)

            if ctx.schema.kind_of(y_col) == "binary":
                y_proba = pipe.predict_proba(X_test)[:, 1]
                metrics["tstr_auc"] = roc_auc_score(y_test, y_proba)
            else:
                y_pred = pipe.predict(X_test)
                metrics["tstr_accuracy"] = accuracy_score(y_test, y_pred)

        else:
            model = Ridge(random_state=ctx.seed)
            pipe = fit_tabular_model(X_syn, y_syn, model)
            y_pred = pipe.predict(X_test)
            metrics["tstr_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))

        return metrics
