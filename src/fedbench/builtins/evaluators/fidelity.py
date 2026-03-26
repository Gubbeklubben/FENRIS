"""
Fidelity evaluators.

Measures how well the statistical properties of the synthetic data match
those of the training data. Includes moment comparison, distribution
similarity tests, categorical total-variation distance, and correlation
matrix comparison.

Federated aggregation notes
----------------------------
* MomentReductionMetricsEvaluator  — **exact** federated aggregation.
  Clients send per-column {real_mean, real_sumsq_dev, real_n, syn_mean,
  syn_sumsq_dev, syn_n}. Server applies Chan's parallel variance formula to
  the real side and takes the synthetic stats from any single client payload.

* DistributionSimilarityMetricsEvaluator — **approximate** federated
  aggregation (same pattern as TSTR utility). Each client computes per-column
  KS, Wasserstein, and |t-stat| against the synthetic data locally and
  reports (scores_dict, n_rows). The server takes a weighted mean per metric.
  The result is a federated proxy, not equivalent to centralized computation,
  because these statistics depend on the global score ranking / joint sample.

* CategoricalTvMeanEvaluator — **exact** federated aggregation.
  Clients send per-column category count dicts; server sums and recomputes
  TV distance from global frequencies.

* CorrFroDiffEvaluator — **centralized-only**. Computing the global
  correlation matrix federally requires sending cross-product matrices that
  reveal joint distribution information. Federated mode is intentionally
  unsupported; see reference guide §3.3.1 / §15.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy

from fedbench.builtins.evaluators._helpers import (
    safe_nanmean,
    sanitize_numeric_df,
    weighted_mean_metrics,
)
from fedbench.core.data import TableSchema
from fedbench.core.eval import Category, Evaluator, LocalEvalContext
from fedbench.core.eval.evalcontext import GlobalEvalContext
from fedbench.core.eval.evaluator import (
    EvaluationMode,
    EvaluatorDescriptor,
    MetricDescriptor,
)
from fedbench.core.logger import log_debug

# ---------------------------------------------------------------------------
# Moment metrics  (mean / std)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MomentReductionResult:
    mean: float
    sumsq_dev: float
    n: int


class MomentReductionMetricsEvaluator(Evaluator):
    """Compare per-column means and standard deviations.

    Federated aggregation: **exact**.

    Local payload
    -------------
    ``dict[str, dict[str, float]]`` with structure::

        {
            col: {
                "real_mean": float, "real_sumsq_dev": float, "real_n": int,
                "syn_mean":  float, "syn_sumsq_dev": float, "syn_n":  int,
            },
            ...
        }

    ``sumsq_dev`` = Σ(x − local_mean)², needed for Chan's parallel variance
    formula.  The synthetic stats are identical on every client (same broadcast
    DataFrame), so ``aggregate`` uses the first non-empty value it sees for the
    synthetic side rather than accumulating it.

    Server-side reduce
    ------------------
    Applies Chan's parallel algorithm to the real-side sufficient statistics
    across clients to recover the global real mean and std, takes the synthetic
    mean and std from any single client's payload, then averages |Δmean| and
    |Δstd| over columns.
    """

    @property
    def id(self) -> str:
        return "moment_reduction_metrics"

    @property
    def metadata(self) -> EvaluatorDescriptor:
        return EvaluatorDescriptor(
            category=Category.FIDELITY,
            eval_mode=EvaluationMode.BOTH,
            metrics=[
                MetricDescriptor("mean_abs_diff"),
                MetricDescriptor("std_abs_diff"),
            ],
        )

    # noinspection PyMethodMayBeStatic
    def _to_numeric_series(self, series: pd.Series) -> pd.Series:
        """Coerce to numeric and drop NaNs, returning a clean pd.Series."""
        return pd.Series(pd.to_numeric(series, errors="coerce")).dropna()

    def _compute(self, df: pd.DataFrame, col: str) -> _MomentReductionResult:
        series = self._to_numeric_series(df[col])
        n = len(series)
        mean = series.mean() if n > 0 else math.nan
        return _MomentReductionResult(
            mean=float(mean),
            sumsq_dev=float(((series - mean) ** 2).sum()) if n > 0 else 0.0,
            n=n,
        )

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        numeric_columns = ctx.schema.numeric_columns(ctx.holdout_df)
        if not numeric_columns:
            return self._nan_result()

        mean_abs_diffs, std_abs_diffs = [], []
        for col in numeric_columns:
            r = self._to_numeric_series(ctx.holdout_df[col])
            s = self._to_numeric_series(ctx.synthetic_df[col])
            if len(r) == 0 or len(s) == 0:
                continue
            mean_abs_diffs.append(abs(r.mean() - s.mean()))
            if len(r) >= 2 and len(s) >= 2:
                std_abs_diffs.append(abs(r.std() - s.std()))

        return {
            "mean_abs_diff": safe_nanmean(mean_abs_diffs),
            "std_abs_diff": safe_nanmean(std_abs_diffs),
        }

    def local_evaluate(
        self, ctx: LocalEvalContext
    ) -> dict[str, dict[str, _MomentReductionResult]]:
        numeric_columns = ctx.schema.numeric_columns(ctx.train_df)
        payload: dict[str, dict[str, _MomentReductionResult]] = {}
        for col in numeric_columns:
            payload[col] = {
                "real": self._compute(ctx.train_df, col),
                "syn": self._compute(ctx.synthetic_df, col),
            }
        return payload

    def aggregate(
        self,
        stats: Iterable[Mapping[str, Mapping[str, _MomentReductionResult]]],
    ) -> dict[str, float]:
        real_acc: dict[str, _MomentReductionResult] = {}
        syn_stats: dict[str, _MomentReductionResult] = {}

        for payload in stats:
            for col, rec in payload.items():
                # Synthetic side: record once per column (identical across clients)
                syn = rec["syn"]
                if col not in syn_stats and syn.n > 0:
                    syn_stats[col] = syn

                # Accumulate real side via Chan's parallel algorithm
                real = rec["real"]
                if real.n <= 0:
                    continue

                if col not in real_acc:
                    real_acc[col] = real
                    continue

                a, b = real_acc[col], real
                n = a.n + b.n
                delta = b.mean - a.mean

                real_acc[col] = _MomentReductionResult(
                    mean=(a.n * a.mean + b.n * b.mean) / n,
                    sumsq_dev=a.sumsq_dev + b.sumsq_dev + delta**2 * a.n * b.n / n,
                    n=n,
                )

        mean_abs_diffs, std_abs_diffs = [], []
        for col, r in real_acc.items():
            s = syn_stats.get(col)
            if s is None:
                continue
            mean_abs_diffs.append(abs(r.mean - s.mean))
            if r.n >= 2 and s.n >= 2:
                r_std = math.sqrt(max(r.sumsq_dev / (r.n - 1), 0.0))
                s_std = math.sqrt(max(s.sumsq_dev / (s.n - 1), 0.0))
                std_abs_diffs.append(abs(r_std - s_std))

        return {
            "mean_abs_diff": safe_nanmean(mean_abs_diffs),
            "std_abs_diff": safe_nanmean(std_abs_diffs),
        }


# ---------------------------------------------------------------------------
# Distribution similarity metrics  (KS / Wasserstein / t-test)
# ---------------------------------------------------------------------------


class DistributionSimilarityMetricsEvaluator(Evaluator):
    """KS statistic, Wasserstein distance, and Welch t-test per numeric column.

    Federated aggregation: **approximate** (same pattern as TSTR utility).

    Each client computes all three metrics locally by comparing its own
    training partition against the synthetic data.  It then reports
    ``(scores_dict, n_rows)`` where ``n_rows`` is the size of the local
    training partition used as the weight in the server-side mean.

    The weighted-mean result is a federated proxy. It is not equivalent to
    running KS/Wasserstein/t-test on the globally pooled real data, because
    all three statistics depend on the joint ranking of values across the
    full sample. This property is not preserved by averaging local scores.
    Results should be labeled accordingly when comparing modes.

    Local payload
    -------------
    ``tuple[dict[str, float], int]`` — ``(scores, n_rows)`` where ``scores``
    has keys ``"ks_mean"``, ``"wasserstein_mean"``, ``"t_stat_mean_abs"``
    and ``n_rows`` is the number of real rows used on this client.
    """

    @property
    def id(self) -> str:
        return "distribution_similarity_metrics"

    @property
    def metadata(self) -> EvaluatorDescriptor:
        return EvaluatorDescriptor(
            category=Category.FIDELITY,
            eval_mode=EvaluationMode.BOTH,
            metrics=[
                MetricDescriptor("ks_mean"),
                MetricDescriptor("wasserstein_mean"),
                MetricDescriptor("t_stat_mean_abs"),
            ],
        )

    # noinspection PyMethodMayBeStatic
    def _compute(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        schema: TableSchema,
    ) -> tuple[dict[str, float], int]:
        numeric_columns = schema.numeric_columns(real_df)
        if not numeric_columns:
            return self._nan_result(), 0

        r_df = sanitize_numeric_df(real_df, numeric_columns)
        s_df = sanitize_numeric_df(syn_df, numeric_columns)
        if r_df.empty or s_df.empty:
            return self._nan_result(), 0

        ks_vals, wasserstein_vals, t_vals = [], [], []
        for col in numeric_columns:
            r = r_df[col]
            s = s_df[col]

            if len(r) == 0 or len(s) == 0:
                continue

            ks_val = scipy.stats.ks_2samp(r, s).statistic
            ks_vals.append(float(ks_val))

            wasserstein_val = scipy.stats.wasserstein_distance(r, s)
            wasserstein_vals.append(float(wasserstein_val))

            t_val = scipy.stats.ttest_ind(r, s, equal_var=False).statistic
            t_vals.append(abs(t_val))

        return {
            "ks_mean": safe_nanmean(ks_vals),
            "wasserstein_mean": safe_nanmean(wasserstein_vals),
            "t_stat_mean_abs": safe_nanmean(t_vals),
        }, len(r_df)

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        scores, _ = self._compute(ctx.holdout_df, ctx.synthetic_df, ctx.schema)
        return scores

    def local_evaluate(self, ctx: LocalEvalContext) -> tuple[dict[str, float], int]:
        return self._compute(ctx.train_df, ctx.synthetic_df, ctx.schema)

    def aggregate(
        self,
        stats: Iterable[tuple[Mapping[str, float], int]],
    ) -> dict[str, float]:
        return weighted_mean_metrics(stats, self.get_metric_keys())


# ---------------------------------------------------------------------------
# Categorical total-variation distance
# ---------------------------------------------------------------------------


class CategoricalTvMeanEvaluator(Evaluator):
    """Mean total-variation distance across categorical columns.

    Federated aggregation: **exact**.

    Local payload
    -------------
    ``dict[str, dict[str, dict[str, int]]]`` with structure::

        {
            col: {
                "real": {category_str: count, ...},
                "syn":  {category_str: count, ...},
            },
            ...
        }

    Server-side reduce
    ------------------
    Sums category counts across clients per column, recomputes empirical
    frequencies from global totals, then derives TV distance.
    """

    @property
    def id(self) -> str:
        return "categorical_tv_mean"

    @property
    def metadata(self) -> EvaluatorDescriptor:
        return EvaluatorDescriptor(
            category=Category.FIDELITY,
            eval_mode=EvaluationMode.BOTH,
            metrics=[
                MetricDescriptor("categorical_tv_mean"),
            ],
        )

    # noinspection PyMethodMayBeStatic
    def _compute(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        schema: TableSchema,
    ) -> dict[str, dict[str, dict[str, int]]]:
        nominal_cols = schema.nominal_columns(real_df)

        def cat_counts(series: pd.Series) -> dict[str, int]:
            return {
                str(k): int(v)
                for k, v in series.fillna("__NA__").value_counts(dropna=False).items()
            }

        return {
            col: {
                "real": cat_counts(real_df[col]),
                "syn": cat_counts(syn_df[col]),
            }
            for col in nominal_cols
        }

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        payload = self._compute(ctx.holdout_df, ctx.synthetic_df, ctx.schema)
        return self.aggregate([payload])

    def local_evaluate(
        self, ctx: LocalEvalContext
    ) -> dict[str, dict[str, dict[str, int]]]:
        return self._compute(ctx.train_df, ctx.synthetic_df, ctx.schema)

    def aggregate(
        self,
        stats: Iterable[dict[str, dict[str, dict[str, int]]]],
    ) -> dict[str, float]:
        acc: dict[str, dict[str, dict[str, int]]] = {}
        for payload in stats:
            for col, sides in payload.items():
                # Synthetic DF is identical on every client — record once only.
                acc.setdefault(col, {"real": {}, "syn": dict(sides["syn"])})
                # Real counts are additive: each client holds a disjoint partition.
                for cat, cnt in sides["real"].items():
                    acc[col]["real"].setdefault(cat, 0)
                    acc[col]["real"][cat] += cnt

        if not acc:
            return self._nan_result()

        tvs = []
        for sides in acc.values():
            nr = sum(sides["real"].values())
            ns = sum(sides["syn"].values())
            if nr == 0 or ns == 0:
                continue

            cats = set(sides["real"]) | set(sides["syn"])
            tv = 0.5 * sum(
                abs(
                    sides["real"].get(c, 0) / nr  # nofmt
                    - sides["syn"].get(c, 0) / ns
                )
                for c in cats
            )
            tvs.append(tv)

        return {"categorical_tv_mean": safe_nanmean(tvs)}


class CorrFroDiffEvaluator(Evaluator):
    """Frobenius norm of the difference between real and synthetic correlation matrices.

    Federated aggregation: **not supported**.
    Computing the global Pearson correlation matrix federally requires each
    client to send O(d²) cross-product sums, which reveals the joint
    distribution of column pairs — a meaningful privacy leakage.  This
    evaluator must therefore be used in centralized mode only; see reference
    guide §3.3.1 and §15.1.
    """

    @property
    def id(self) -> str:
        return "corr_fro_diff"

    @property
    def metadata(self) -> EvaluatorDescriptor:
        return EvaluatorDescriptor(
            category=Category.FIDELITY,
            eval_mode=EvaluationMode.CENTRALIZED,
            metrics=[
                MetricDescriptor("corr_fro_diff"),
            ],
        )

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        numeric_columns = ctx.schema.numeric_columns(ctx.holdout_df)
        if len(numeric_columns) < 2:
            return self._nan_result()

        r = ctx.holdout_df[numeric_columns]
        s = ctx.synthetic_df[numeric_columns]

        # Filter columns jointly to ensure that both matrices have the same column set.
        # A constant column produces NaN correlations on its own side and would bias
        # the computation if included.
        non_constant = [
            col
            for col in numeric_columns
            if r[col].nunique(dropna=True) > 1 and s[col].nunique(dropna=True) > 1
        ]
        if len(non_constant) < 2:
            return self._nan_result()

        r_corr = r[non_constant].corr().fillna(0.0)
        s_corr = s[non_constant].corr().fillna(0.0)

        diff = r_corr.values - s_corr.values
        return {
            "corr_fro_diff": float(np.linalg.norm(diff, ord="fro")),
        }

    def local_evaluate(self, ctx: LocalEvalContext) -> dict[str, float]:
        log_debug(
            "CorrFroDiffEvaluator",
            "CorrFroDiffEvaluator does not support federated evaluation. "
            "Use global_evaluate (centralized mode) only.",
        )
        return self._nan_result()

    def aggregate(self, stats: Iterable[Mapping[str, float]]) -> dict[str, float]:
        log_debug(
            "CorrFroDiffEvaluator",
            "CorrFroDiffEvaluator does not support federated aggregation. "
            "Use global_evaluate (centralized mode) only.",
        )
        return self._nan_result()
