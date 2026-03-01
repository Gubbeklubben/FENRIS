import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from fedbench.core.eval import Evaluator, EvalContext
from fedbench.util.metrics import get_quasi_identifiers, get_schema_columns, fit_tabular_model, canonical_row_hash
from fedbench.util.parsing import to_snake_case

MAX_MIA_SYNTHETIC = 5000
DEFAULT_MIA_K = 1000


# ============================================================
# 10.4.1 Direct overlap / memorization diagnostics
# ============================================================

class DirectOverlapDiagnosticEvaluator(Evaluator):
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


# ============================================================
# 10.4.2 Membership inference attack (nearest neighbor)
# ============================================================

class MIANearestNeighborAttackEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        train, test, syn = ctx.train_df, ctx.test_df, ctx.synthetic_df

        K = min(DEFAULT_MIA_K, len(train), len(test))
        if K == 0:
            return {
                "mia_auc": math.nan,
                "mia_accuracy": math.nan,
                "mia_advantage": math.nan
            }

        members = train.sample(n=K, random_state=ctx.seed)
        nonmembers = test.sample(n=K, random_state=ctx.seed)

        X = pd.concat([members, nonmembers], ignore_index=True)
        y = np.array([1] * K + [0] * K)

        # Subsample synthetic data for tractability
        if len(syn) > MAX_MIA_SYNTHETIC:
            syn = syn.sample(n=MAX_MIA_SYNTHETIC, random_state=ctx.seed)

        numeric_cols, categorical_cols = get_schema_columns(ctx)

        # Scale numeric columns
        if numeric_cols:
            scaler = MinMaxScaler()
            scaler.fit(train[numeric_cols])
            X_num = scaler.transform(X[numeric_cols])
            syn_num = scaler.transform(syn[numeric_cols])
        else:
            X_num = np.zeros((len(X), 0))
            syn_num = np.zeros((len(syn), 0))

        # Categorical arrays
        if categorical_cols:
            X_cat = X[categorical_cols].to_numpy()
            syn_cat = syn[categorical_cols].to_numpy()
        else:
            X_cat = np.zeros((len(X), 0))
            syn_cat = np.zeros((len(syn), 0))

        # Compute distances vectorized
        dists = np.zeros((len(X), len(syn)))

        if numeric_cols:
            dists += np.linalg.norm(X_num[:, None, :] - syn_num[None, :, :], axis=2)

        if categorical_cols:
            dists += (X_cat[:, None, :] != syn_cat[None, :, :]).mean(axis=2)

        # Minimum distance per candidate
        min_dists = dists.min(axis=1)
        scores = -min_dists

        threshold = np.median(scores)
        return {
            "mia_auc": roc_auc_score(y, scores),
            "mia_accuracy": accuracy_score(y, scores > threshold),
            "mia_advantage": np.mean(scores[y == 1] > threshold) - np.mean(scores[y == 0] > threshold)
        }


# ============================================================
# 10.4.3 Attribute inference attack (supervised)
# ============================================================

class AIASupervisedAttackEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
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
                    random_state=ctx.seed
                )
                pipe = fit_tabular_model(X_syn, y_syn, model)

                if len(np.unique(y_syn)) == 2:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    metrics[f"aia_auc.{sensitive_column_normalized}"] = roc_auc_score(y_test, y_proba)

                y_pred = pipe.predict(X_test)

                metrics[f"aia_accuracy.{sensitive_column_normalized}"] = accuracy_score(y_test, y_pred)

            else:  # regression
                model = Ridge(random_state=ctx.seed)
                pipe = fit_tabular_model(X_syn, y_syn, model)
                y_pred = pipe.predict(X_test)
                metrics[f"aia_rmse.{sensitive_column_normalized}"] = math.sqrt(mean_squared_error(y_test, y_pred))

        return metrics or {"aia": math.nan}