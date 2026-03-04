from __future__ import annotations

import math
from abc import ABC
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from fedbench.core.eval import EvalContext, Evaluator
from fedbench.util.metrics import get_schema_columns


def _safe_nanmean(values: list[float]) -> float:
    """Like ``np.nanmean`` but returns ``nan`` silently for all-NaN inputs.

    ``np.nanmean`` emits a ``RuntimeWarning: Mean of empty slice`` when every
    element is NaN.  This helper avoids that by returning ``math.nan``
    directly when no finite values remain.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    return float(np.mean(finite)) if finite.size else math.nan


class MomentReductionMetricsEvaluator(Evaluator, ABC):
    def evaluate(self, ctx: EvalContext) -> Mapping[str, float]:
        numeric_columns, _ = get_schema_columns(ctx)
        if not numeric_columns:
            return {
                "mean_abs_diff": math.nan,
                "std_abs_diff": math.nan,
            }

        mean_abs_diff = []
        std_abs_diff = []

        for col in numeric_columns:
            r = pd.to_numeric(ctx.train_df[col], errors="coerce")
            s = pd.to_numeric(ctx.synthetic_df[col], errors="coerce")

            mean_abs_diff.append(abs(r.mean() - s.mean()))
            std_abs_diff.append(abs(r.std() - s.std()))

        return {
            "mean_abs_diff": _safe_nanmean(mean_abs_diff),
            "std_abs_diff": _safe_nanmean(std_abs_diff),
        }


class DistributionSimilarityMetricsEvaluator(Evaluator, ABC):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        metrics = {
            "ks_mean": math.nan,
            "wasserstein_mean": math.nan,
            "t_stat_mean_abs": math.nan,
        }
        numeric_columns, _ = get_schema_columns(ctx)
        if not numeric_columns:
            return metrics

        ks = []
        wasserstein = []
        t_stats = []

        for col in numeric_columns:
            r = ctx.train_df[col].astype(float).dropna()
            s = ctx.synthetic_df[col].astype(float).dropna()

            if len(r) == 0 or len(s) == 0:
                continue

            ks_stat, _ = stats.ks_2samp(r, s)
            ks.append(ks_stat)

            wasserstein_distance = stats.wasserstein_distance(r, s)
            wasserstein.append(wasserstein_distance)

            t_stat, _ = stats.ttest_ind(r, s, equal_var=False)
            t_stats.append(abs(t_stat))

        if ks:
            metrics["ks_mean"] = _safe_nanmean(ks)
        if wasserstein:
            metrics["wasserstein_mean"] = _safe_nanmean(wasserstein)
        if t_stats:
            metrics["t_stat_mean_abs"] = _safe_nanmean(t_stats)

        return metrics


class CategoricalTvMeanEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        _, categorical_columns = get_schema_columns(ctx)
        if not categorical_columns:
            return {
                "categorical_tv_mean": math.nan
            }

        tvs = []
        for col in categorical_columns:
            pr = ctx.train_df[col].fillna("__NA__").value_counts(normalize=True)
            ps = ctx.synthetic_df[col].fillna("__NA__").value_counts(normalize=True)

            vals = set(pr.index).union(ps.index)
            tv = 0.5 * sum(abs(pr.get(v, 0) - ps.get(v, 0)) for v in vals)
            tvs.append(tv)

        return {
            "categorical_tv_mean": float(np.mean(tvs))
        }


class CorrFroDiffEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        numeric_columns, _ = get_schema_columns(ctx)
        if len(numeric_columns) < 2:
            return {
                "corr_fro_diff": math.nan
            }

        def safe_corr(df: pd.DataFrame) -> pd.DataFrame:
            # Drop zero-variance columns
            non_constant = df.columns[df.nunique(dropna=True) > 1]
            corr = df[non_constant].corr()
            return corr.fillna(0.0)

        r_corr = safe_corr(ctx.train_df[numeric_columns])
        s_corr = safe_corr(ctx.synthetic_df[numeric_columns])

        # Align matrices (important if columns dropped differently)
        common = r_corr.index.intersection(s_corr.index)
        diff = r_corr.loc[common, common].values - s_corr.loc[common, common].values

        return {
            "corr_fro_diff": float(np.linalg.norm(diff, ord="fro"))
        }
