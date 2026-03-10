"""
Privacy evaluators.

Includes three complementary privacy diagnostics:

* **Direct overlap** — detects exact or near-exact row memorisation.
* **Membership inference attack (MIA)** — nearest-neighbour shadow-model
  attack estimating whether a record was used during training.
* **Attribute inference attack (AIA)** — supervised attack that tries to
  infer a sensitive attribute from quasi-identifier columns.
"""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from fedbench.core.eval import EvalContext, Evaluator
from fedbench.util.metrics import (
    canonical_row_hash,
    fit_tabular_model,
    get_numeric_columns,
    get_quasi_identifiers,
    sanitize_numeric_df,
)
from fedbench.util.parsing import to_snake_case

MAX_MIA_SYNTHETIC = 5000
DEFAULT_MIA_K = 1000


class DirectOverlapDiagnosticEvaluator(Evaluator):
    """Detect exact and near-exact row memorisation in the synthetic dataset.

    Computes exact-match rates and partial-match rates (top-1/2/3 most-unique
    columns) between the synthetic data and the training set.
    """

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        train, syn = ctx.train_df, ctx.synthetic_df
        common = sorted(set(train.columns) & set(syn.columns))

        if not common:
            return {
                "exact_row_match_rate_train": math.nan,
                "exact_row_match_any": math.nan,
                "partial_match_rate_top1": math.nan,
                "partial_match_rate_top2": math.nan,
                "partial_match_rate_top3": math.nan,
                "partial_match_any": math.nan,
            }

        H_train = set(canonical_row_hash(train[common]))
        H_syn = canonical_row_hash(syn[common])

        exact_rate = float(np.mean([h in H_train for h in H_syn]))

        # Identify columns by uniqueness ratio
        excluded = set(ctx.sensitive_columns or [])
        if ctx.target_column:
            excluded.add(ctx.target_column)

        candidates = [c for c in common if c not in excluded] or common
        ranked = sorted(
            candidates,
            key=lambda c: train[c].nunique() / len(train),
            reverse=True,
        )

        partial_rates = {}
        for k in (1, 2, 3):
            cols = ranked[:k]
            Ht = set(canonical_row_hash(train[cols]))
            Hs = canonical_row_hash(syn[cols])
            partial_rates[k] = float(np.mean([h in Ht for h in Hs]))

        return {
            "exact_row_match_rate_train": exact_rate,
            "exact_row_match_any": float(exact_rate > 0),
            "partial_match_rate_top1": partial_rates[1],
            "partial_match_rate_top2": partial_rates[2],
            "partial_match_rate_top3": partial_rates[3],
            "partial_match_any": float(any(v > 0 for v in partial_rates.values())),
        }


class MIANearestNeighborAttackEvaluator(Evaluator):
    """Nearest-neighbour membership inference attack.

    Labels training records as *members* and held-out records as
    *non-members*, then scores each record by its negative distance to the
    nearest synthetic sample. Reports AUC, accuracy, and advantage.
    """

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        nan_result = {
            "mia_auc": math.nan,
            "mia_accuracy": math.nan,
            "mia_advantage": math.nan,
        }

        K = min(DEFAULT_MIA_K, len(ctx.train_df), len(ctx.test_df))
        if K == 0 or len(ctx.synthetic_df) == 0:
            return nan_result

        numeric_cols = get_numeric_columns(ctx.train_df, ctx.schema)
        if not numeric_cols:
            return nan_result

        rt = sanitize_numeric_df(ctx.train_df, numeric_cols)
        rh = sanitize_numeric_df(ctx.test_df, numeric_cols)
        sx = sanitize_numeric_df(ctx.synthetic_df, numeric_cols)

        if rt.empty or rh.empty or sx.empty:
            return nan_result

        k_train = min(K, len(rt))
        k_test = min(K, len(rh))

        members = rt.sample(n=k_train, random_state=ctx.seed)
        nonmembers = rh.sample(n=k_test, random_state=ctx.seed)

        X = pd.concat([members, nonmembers], ignore_index=True)
        y = np.array([1] * k_train + [0] * k_test)

        syn_mat = sx.to_numpy(dtype=float)
        syn_min = syn_mat.min(axis=0)
        syn_rng = syn_mat.max(axis=0) - syn_min
        syn_rng[syn_rng == 0] = 1.0

        syn_norm = (syn_mat - syn_min) / syn_rng

        def nn_dist(x: np.ndarray) -> float:
            x = (x - syn_min) / syn_rng
            d2 = np.sum((syn_norm - x) ** 2, axis=1)
            return float(np.sqrt(np.min(d2)))

        dists = np.array([nn_dist(v) for v in X.to_numpy(dtype=float)])
        scores = -dists

        finite = scores[np.isfinite(scores)]
        if len(finite) == 0:
            return nan_result
        scores = np.where(np.isfinite(scores), scores, finite.min())

        threshold = np.median(scores)

        return {
            "mia_auc": roc_auc_score(y, scores),
            "mia_accuracy": accuracy_score(y, scores > threshold),
            "mia_advantage": (
                np.mean(scores[y == 1] > threshold)
                - np.mean(scores[y == 0] > threshold)
            ),
        }


class AIASupervisedAttackEvaluator(Evaluator):
    """Supervised attribute inference attack.

    Trains a classifier (or regressor) on the synthetic data to predict a
    sensitive attribute from quasi-identifier columns, then evaluates on
    real held-out data. Reports accuracy, AUC, and RMSE per sensitive column.
    """

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        nan_result = {
            "aia_accuracy": math.nan,
            "aia_auc": math.nan,
            "aia_rmse": math.nan,
        }

        train, test, syn = ctx.train_df, ctx.test_df, ctx.synthetic_df
        all_columns = set(train.columns)

        metrics: dict[str, float] = {}

        for sensitive_column in ctx.sensitive_columns or []:
            sensitive_column_normalized = to_snake_case(sensitive_column)

            metrics[f"aia_accuracy.{sensitive_column_normalized}"] = math.nan
            metrics[f"aia_auc.{sensitive_column_normalized}"] = math.nan
            metrics[f"aia_rmse.{sensitive_column_normalized}"] = math.nan

            quasi_ids = get_quasi_identifiers(
                all_columns,
                sensitive_column,
                ctx.target_column,
            )

            if not quasi_ids:
                continue

            X_syn = syn[quasi_ids]
            y_syn = syn[sensitive_column]

            X_test = test[quasi_ids]
            y_test = test[sensitive_column]

            if ctx.schema.kind_of(sensitive_column) in ["binary", "categorical"]:
                model = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=ctx.seed,
                )
                pipe = fit_tabular_model(X_syn, y_syn, model)

                if len(np.unique(y_syn)) == 2:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    metrics[f"aia_auc.{sensitive_column_normalized}"] = roc_auc_score(
                        y_test, y_proba
                    )

                y_pred = pipe.predict(X_test)

                metrics[f"aia_accuracy.{sensitive_column_normalized}"] = accuracy_score(
                    y_test, y_pred
                )

            else:  # regression
                model = Ridge(random_state=ctx.seed)
                pipe = fit_tabular_model(X_syn, y_syn, model)
                y_pred = pipe.predict(X_test)
                metrics[f"aia_rmse.{sensitive_column_normalized}"] = math.sqrt(
                    mean_squared_error(y_test, y_pred)
                )

        return metrics or nan_result
