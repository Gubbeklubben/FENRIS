"""
Fidelity evaluators.

Measures how well the statistical properties of the synthetic data match
those of the training data. Includes moment comparison, distribution
similarity tests, categorical total-variation distance, and correlation
matrix comparison.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from fedbench.core.eval import EvalContext, Evaluator
from fedbench.util.metrics import get_schema_columns, safe_nanmean, sanitize_numeric_df


class MomentReductionMetricsEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> Mapping[str, float]:
        nan_result = {
            "mean_abs_diff": math.nan,
            "std_abs_diff": math.nan,
        }

        numeric_columns, _ = get_schema_columns(ctx)
        if not numeric_columns:
            return nan_result

        r_df = sanitize_numeric_df(ctx.train_df, numeric_columns)
        s_df = sanitize_numeric_df(ctx.synthetic_df, numeric_columns)

        if r_df.empty or s_df.empty:
            return nan_result

        mean_abs_diff = []
        std_abs_diff = []

        for col in numeric_columns:
            r = r_df[col]
            s = s_df[col]

            mean_abs_diff.append(abs(r.mean() - s.mean()))
            std_abs_diff.append(abs(r.std() - s.std()))

        return {
            "mean_abs_diff": safe_nanmean(mean_abs_diff),
            "std_abs_diff": safe_nanmean(std_abs_diff),
        }


class DistributionSimilarityMetricsEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        nan_result = {
            "ks_mean": math.nan,
            "wasserstein_mean": math.nan,
            "t_stat_mean_abs": math.nan,
        }

        numeric_columns, _ = get_schema_columns(ctx)
        if not numeric_columns:
            return nan_result

        r_df = sanitize_numeric_df(ctx.train_df, numeric_columns)
        s_df = sanitize_numeric_df(ctx.synthetic_df, numeric_columns)

        if r_df.empty or s_df.empty:
            return nan_result

        ks = []
        wasserstein = []
        t_stats = []

        for col in numeric_columns:
            r = r_df[col]
            s = s_df[col]

            if len(r) == 0 or len(s) == 0:
                continue

            ks_stat, _ = stats.ks_2samp(r, s)
            ks.append(float(ks_stat))

            wasserstein_distance = stats.wasserstein_distance(r, s)
            wasserstein.append(float(wasserstein_distance))

            ttest_res = stats.ttest_ind(r, s, equal_var=False)
            t_stats.append(abs(ttest_res.statistic))

        return {
            "ks_mean": safe_nanmean(ks),
            "wasserstein_mean": safe_nanmean(wasserstein),
            "t_stat_mean_abs": safe_nanmean(t_stats),
        }


class CategoricalTvMeanEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        _, categorical_columns = get_schema_columns(ctx)
        if not categorical_columns:
            return {
                "categorical_tv_mean": math.nan,
            }

        tvs = []
        for col in categorical_columns:
            pr = ctx.train_df[col].fillna("__NA__").value_counts(normalize=True)
            ps = ctx.synthetic_df[col].fillna("__NA__").value_counts(normalize=True)

            vals = set(pr.index).union(ps.index)
            tv = 0.5 * sum(abs(pr.get(v, 0) - ps.get(v, 0)) for v in vals)
            tvs.append(tv)

        return {
            "categorical_tv_mean": float(np.mean(tvs)),
        }


class CorrFroDiffEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        numeric_columns, _ = get_schema_columns(ctx)
        if len(numeric_columns) < 2:
            return {
                "corr_fro_diff": math.nan,
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
            "corr_fro_diff": float(np.linalg.norm(diff, ord="fro")),
        }
