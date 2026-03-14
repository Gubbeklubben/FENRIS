"""
Privacy evaluators.

Includes three complementary privacy diagnostics:

* **Direct overlap** — detects exact or near-exact row memorization.
* **Membership inference attack (MIA)** — nearest-neighbor shadow-model
  attack estimating whether a record was used during training.
* **Attribute inference attack (AIA)** — supervised attack that tries to
  infer a sensitive attribute from quasi-identifier columns.

Federated aggregation notes
----------------------------
DirectOverlapDiagnosticEvaluator — **exact** federated aggregation.
  Each client hashes its own training rows, counts hash collisions against
  the synthetic DataFrame, and reports only (match_count, n_train, n_syn).
  Raw data and hashes never leave the client.
  Server global rate = Σ match_counts / n_syn  (synthetic set is identical
  for all clients so n_syn is the same everywhere).
  global_evaluate returns NaN — overlap against a server holdout is a
  structural false negative by construction (see reference guide §16.5).

MIANearestNeighborAttackEvaluator — **approximate** federated aggregation.
  global_evaluate requires a CentralizedEvalContext because it needs access
  to real client training data (members) alongside the holdout (non-members)
  to produce a meaningful AUC; using only a server-held holdout cannot detect
  client-side memorization.
  In federated mode each client computes a local AUC from its own
  members/non-members and reports (local_auc, n_pos, n_neg).
  Server produces a weighted-mean proxy AUC (not equivalent to centralized).
  See reference guide §3.3.3 and §15.3.2 for the exactness caveat.

AIASupervisedAttackEvaluator — **exact** federated aggregation (per
  sensitive column).  The attacker model is trained on the same synthetic
  data everywhere; clients evaluate it on their local test splits and report
  (accuracy, n_test) [and optionally auc / rmse].  Server computes a
  weighted mean, which is mathematically equivalent to centralized because
  all clients run the same attacker.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from fedbench.core.data import TableSchema
from fedbench.core.eval import Evaluator, LocalEvalContext
from fedbench.core.eval.evalcontext import CentralizedEvalContext, GlobalEvalContext
from fedbench.core.logger import log_debug
from fedbench.util.metrics import (
    canonical_row_hash,
    fit_tabular_model,
    get_numeric_columns,
    get_quasi_identifiers,
    sanitize_numeric_df,
    weighted_mean,
)
from fedbench.util.parsing import to_snake_case

# ---------------------------------------------------------------------------
# Direct overlap evaluator
# ---------------------------------------------------------------------------


class DirectOverlapDiagnosticEvaluator(Evaluator):
    """Detect exact and near-exact row memorization in the synthetic dataset.

    Computes exact-match rates and partial-match rates (top-1/2/3 most-unique
    columns) between the synthetic data and the training set.

    Federated aggregation: **exact**.

    Local payload
    -------------
    ``None`` if not valid, otherwise ``dict`` with keys:

    * ``"exact_matches"``   — int, number of synthetic rows whose hash
      appears in the client's training-row hash set.
    * ``"partial_matches"`` — ``dict[int, int]`` mapping k → match count
      for k ∈ {1, 2, 3}.
    * ``"n_syn"``           — int, total number of synthetic rows evaluated.

    Server-side reduce
    ------------------
    Global exact rate = Σ exact_matches / n_syn.
    Partial rates computed analogously.  n_syn is identical across clients
    (same synthetic DF broadcast to all), so any client's value is used.
    """

    # noinspection PyMethodMayBeStatic
    def _nan_result(self) -> dict[str, float]:
        return {
            "exact_row_match_rate_train": math.nan,
            "exact_row_match_any": math.nan,
            "partial_match_rate_top1": math.nan,
            "partial_match_rate_top2": math.nan,
            "partial_match_rate_top3": math.nan,
            "partial_match_any": math.nan,
        }

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        """Overlap must be checked against client training records (federated mode).

        Centralized evaluation using a server-held holdout produces a structural
        false negative by construction — memorized training records cannot appear
        in a holdout that is disjoint from D_train. See reference guide §16.5.
        """
        log_debug(
            "DirectOverlapDiagnosticEvaluator",
            "global_evaluate is not meaningful for overlap diagnostics: the server "
            "holdout is disjoint from D_train by construction. Use federated mode "
            "(local_evaluate + aggregate) instead. Returning NaN. See §16.5.",
        )
        return self._nan_result()

    def local_evaluate(self, ctx: LocalEvalContext) -> dict[str, Any] | None:
        """Compute exact and partial match counts between train_df and syn_df."""
        common = sorted(set(ctx.train_df.columns) & set(ctx.synthetic_df.columns))
        if not common:
            return None

        H_train = set(canonical_row_hash(ctx.train_df[common]))
        H_syn = canonical_row_hash(ctx.synthetic_df[common])
        exact_matches = int(sum(h in H_train for h in H_syn))

        excluded = set(ctx.sensitive_columns or [])
        if ctx.target_column:
            excluded.add(ctx.target_column)

        candidates = [c for c in common if c not in excluded] or common
        ranked = sorted(
            candidates,
            key=lambda c: ctx.train_df[c].nunique() / max(len(ctx.train_df), 1),
            reverse=True,
        )

        partial_matches: dict[int, int] = {}
        for k in (1, 2, 3):
            cols = ranked[:k]
            Ht = set(canonical_row_hash(ctx.train_df[cols]))
            Hs = canonical_row_hash(ctx.synthetic_df[cols])
            partial_matches[k] = int(sum(h in Ht for h in Hs))

        return {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "n_syn": len(H_syn),
        }

    def aggregate(
        self,
        stats: Iterable[dict[str, Any] | None],
    ) -> dict[str, float]:
        valid = [s for s in stats if s]
        if not valid:
            return self._nan_result()

        # n_syn is identical across clients (same synthetic DF)
        n_syn = valid[0]["n_syn"]
        if not n_syn:
            return self._nan_result()

        exact_rate = sum(s["exact_matches"] for s in valid) / n_syn
        partial_rates = {
            k: sum(s["partial_matches"].get(k, 0) for s in valid) / n_syn
            for k in (1, 2, 3)
        }

        return {
            "exact_row_match_rate_train": exact_rate,
            "exact_row_match_any": float(exact_rate > 0),
            "partial_match_rate_top1": partial_rates[1],
            "partial_match_rate_top2": partial_rates[2],
            "partial_match_rate_top3": partial_rates[3],
            "partial_match_any": float(any(v > 0 for v in partial_rates.values())),
        }


# ---------------------------------------------------------------------------
# MIA — nearest-neighbour attack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MIAResult:
    mia_auc: float
    mia_accuracy: float
    mia_advantage: float
    K: int


class MIANearestNeighborAttackEvaluator(Evaluator):
    """Nearest-neighbor membership inference attack.

    Labels training records as *members* and held-out records as
    *non-members*, then scores each record by its negative distance to the
    nearest synthetic sample. Reports AUC, accuracy, and advantage.

    Centralized mode (``global_evaluate``)
    ---------------------------------------
    Requires a :class:`~fedbench.core.eval.evalcontext.CentralizedEvalContext`
    so that the evaluator can sample true members from ``ctx.client_train_df``
    and non-members from ``ctx.holdout_df``.  Using only a server-side holdout
    for both roles cannot detect client-side memorization, which is the
    primary threat the MIA is designed to surface.

    Federated mode
    ---------------
    ``local_evaluate`` runs the NN attack locally per client using its own
    training partition as members and its local test split as non-members.
    The server aggregates via a weighted mean AUC (approximate; see
    reference guide §15.3.2).
    """

    DEFAULT_MIA_K = 1000

    def _compute(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        schema: TableSchema,
        seed: int,
    ) -> _MIAResult:
        """Core NN-MIA logic shared by centralized and federated paths."""
        nan_result = _MIAResult(
            mia_auc=math.nan,
            mia_accuracy=math.nan,
            mia_advantage=math.nan,
            K=0,
        )

        if len(syn_df) == 0:
            return nan_result

        numeric_cols = get_numeric_columns(train_df, schema)
        if not numeric_cols:
            return nan_result

        members_pool = sanitize_numeric_df(train_df, numeric_cols)
        nonmembers_pool = sanitize_numeric_df(test_df, numeric_cols)
        sx = sanitize_numeric_df(syn_df, numeric_cols)

        if members_pool.empty or nonmembers_pool.empty or sx.empty:
            return nan_result

        K = min(self.DEFAULT_MIA_K, len(members_pool), len(nonmembers_pool))
        if K == 0:
            return nan_result

        members = members_pool.sample(n=K, random_state=seed)
        nonmembers = nonmembers_pool.sample(n=K, random_state=seed)

        X = pd.concat([members, nonmembers], ignore_index=True)
        y = np.array([1] * len(members) + [0] * len(nonmembers))

        syn_mat = sx.to_numpy(dtype=float)
        syn_min = syn_mat.min(axis=0)
        syn_rng = syn_mat.max(axis=0) - syn_min
        syn_rng[syn_rng == 0] = 1.0
        syn_norm = (syn_mat - syn_min) / syn_rng

        def nn_dist(x: np.ndarray) -> float:
            x_norm = (x - syn_min) / syn_rng
            d2 = np.sum((syn_norm - x_norm) ** 2, axis=1)
            return float(np.sqrt(np.min(d2)))

        dists = np.array([nn_dist(v) for v in X.to_numpy(dtype=float)])
        scores = -dists

        finite = scores[np.isfinite(scores)]
        if len(finite) == 0:
            return nan_result
        scores = np.where(np.isfinite(scores), scores, finite.min())

        threshold = np.median(scores)
        return _MIAResult(
            K=K,
            mia_auc=roc_auc_score(y, scores),
            mia_accuracy=accuracy_score(y, scores > threshold),
            mia_advantage=float(
                np.mean(scores[y == 1] > threshold)
                - np.mean(scores[y == 0] > threshold)
            ),
        )

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        """Centralized MIA — requires CentralizedEvalContext.

        Members are sampled from ``ctx.client_train_df``; non-members from
        ``ctx.holdout_df``.  This is the recommended mode for MIA per the
        reference guide (§15.3.2).
        """

        if not isinstance(ctx, CentralizedEvalContext):
            log_debug(
                "MIANearestNeighborAttackEvaluator",
                "global_evaluate requires CentralizedEvalContext to access client "
                "training data for membership sampling. Returning NaN.",
            )
            return {
                "mia_auc": math.nan,
                "mia_accuracy": math.nan,
                "mia_advantage": math.nan,
            }

        result = self._compute(
            ctx.client_train_df,
            ctx.holdout_df,
            ctx.synthetic_df,
            ctx.schema,
            ctx.seed,
        )
        return {
            "mia_auc": result.mia_auc,
            "mia_accuracy": result.mia_accuracy,
            "mia_advantage": result.mia_advantage,
        }

    def local_evaluate(self, ctx: LocalEvalContext) -> tuple[float, int]:
        """Return ``(local_auc, K)`` for weighted-mean aggregation."""
        result = self._compute(
            ctx.train_df,
            ctx.test_df,
            ctx.synthetic_df,
            ctx.schema,
            ctx.seed,
        )
        return result.mia_auc, result.K

    def aggregate(self, stats: Iterable[tuple[float, int]]) -> dict[str, float]:
        """Weighted-mean AUC across clients (approximate federated proxy)."""
        pairs: list[tuple[float, int]] = [
            (auc, n)  # nofmt
            for auc, n in stats
            if not math.isnan(auc) and n > 0
        ]
        return {
            "mia_auc": weighted_mean(pairs),
            # accuracy and advantage cannot be meaningfully aggregated
            # across clients without pooling raw scores
            "mia_accuracy": math.nan,
            "mia_advantage": math.nan,
        }


# ---------------------------------------------------------------------------
# AIA — supervised attribute inference attack, shared scoring helper
# ---------------------------------------------------------------------------


class AIASupervisedAttackEvaluator(Evaluator):
    """Supervised attribute inference attack.

    Trains a classifier (or regressor) on the synthetic data to predict a
    sensitive attribute from quasi-identifier columns, then evaluates on
    real held-out data. Reports accuracy, AUC, and RMSE per sensitive column.

    Federated aggregation: **exact** (same attacker model on all clients).

    Local payload
    -------------
    ``dict[str, dict[str, float | int]]`` mapping each sensitive-column key
    to ``{"accuracy": float, "auc": float, "rmse": float, "n_test": int}``.
    Values that cannot be computed are ``math.nan``.

    Server-side reduce
    ------------------
    Weighted mean per metric key, weighted by ``n_test``.
    """

    # noinspection PyMethodMayBeStatic
    def _nan_result(self) -> dict[str, float]:
        return {
            "aia_accuracy": math.nan,
            "aia_auc": math.nan,
            "aia_rmse": math.nan,
        }

    # noinspection PyMethodMayBeStatic
    def _compute_column(
        self,
        test_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        target_column: str | None,
        sensitive_column: str,
        schema: Any,
        seed: int,
    ) -> dict[str, float | int]:
        """Fit an attacker on syn_df and evaluate on (X_test, y_test).

        Returns a dict with keys ``"accuracy"``, ``"auc"``, ``"rmse"``
        (NaN for whichever metrics are not applicable to the column kind).
        """

        entry: dict[str, float] = {
            "accuracy": math.nan,
            "auc": math.nan,
            "rmse": math.nan,
            "n_test": 0,
        }

        all_columns = set(test_df.columns)
        quasi_ids = get_quasi_identifiers(all_columns, sensitive_column, target_column)
        if (
            not quasi_ids
            or sensitive_column not in test_df.columns
            or sensitive_column not in syn_df.columns
        ):
            return entry

        X_test = test_df[quasi_ids]
        y_test = test_df[sensitive_column]
        X_syn = syn_df[quasi_ids]
        y_syn = syn_df[sensitive_column]
        entry["n_test"] = len(test_df)

        try:
            if schema.kind_of(sensitive_column) in ["binary", "categorical"]:
                model = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed,
                )
                pipe = fit_tabular_model(X_syn, y_syn, model)
                y_pred = pipe.predict(X_test)
                entry["accuracy"] = accuracy_score(y_test, y_pred)
                if len(np.unique(y_syn)) == 2:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    entry["auc"] = roc_auc_score(y_test, y_proba)
            else:
                model = Ridge(random_state=seed)
                pipe = fit_tabular_model(X_syn, y_syn, model)
                y_pred = pipe.predict(X_test)
                entry["rmse"] = math.sqrt(mean_squared_error(y_test, y_pred))
        except Exception:
            pass  # leave values as nan

        return entry

    # noinspection PyMethodMayBeStatic
    def _compute(
        self,
        test_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        target_column: str | None,
        sensitive_columns: Iterable[str] | None,
        schema: Any,
        seed: int,
    ) -> dict[str, dict[str, float | int]]:
        payload: dict[str, dict[str, float | int]] = {}

        for sensitive_column in sensitive_columns or []:
            key = to_snake_case(sensitive_column)

            payload[key] = self._compute_column(
                test_df,
                syn_df,
                target_column,
                sensitive_column,
                schema,
                seed,
            )

        return payload

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        results = self._compute(
            ctx.holdout_df,
            ctx.synthetic_df,
            ctx.target_column,
            ctx.sensitive_columns,
            ctx.schema,
            ctx.seed,
        )
        metrics: dict[str, float] = {}

        for key, scores in results.items():
            metrics[f"aia_accuracy.{key}"] = scores["accuracy"]
            metrics[f"aia_auc.{key}"] = scores["auc"]
            metrics[f"aia_rmse.{key}"] = scores["rmse"]

        return metrics or self._nan_result()

    def local_evaluate(
        self, ctx: LocalEvalContext
    ) -> dict[str, dict[str, float | int]]:
        return self._compute(
            ctx.test_df,
            ctx.synthetic_df,
            ctx.target_column,
            ctx.sensitive_columns,
            ctx.schema,
            ctx.seed,
        )

    def aggregate(
        self,
        stats: Iterable[dict[str, dict[str, float | int]]],
    ) -> dict[str, float]:
        acc: dict[str, list[tuple[float, int]]] = {}

        for payload in stats:
            for col_key, entry in payload.items():
                n = int(entry.get("n_test", 0))
                for metric in ("accuracy", "auc", "rmse"):
                    full_key = f"aia_{metric}.{col_key}"
                    acc.setdefault(full_key, [])
                    v = entry.get(metric, math.nan)
                    if not math.isnan(float(v)) and n > 0:
                        acc[full_key].append((float(v), n))

        if not acc:
            return self._nan_result()

        return {
            key: weighted_mean(pairs)  # nofmt
            for key, pairs in acc.items()
        }
