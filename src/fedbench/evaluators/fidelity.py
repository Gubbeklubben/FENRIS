from __future__ import annotations

from abc import ABC
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from fedbench.core.eval import EvalContext, Evaluator
from fedbench.util.metrics import get_schema_columns


class MomentReductionMetricsEvaluator(Evaluator, ABC):
    def evaluate(self, ctx: EvalContext) -> Mapping[str, float]:
        numeric_columns, _ = get_schema_columns(ctx)
        if not numeric_columns:
            return {}

        mean_abs_diff = []
        std_abs_diff = []

        for col in numeric_columns:
            r = pd.to_numeric(ctx.train_df[col], errors="coerce")
            s = pd.to_numeric(ctx.synthetic_df[col], errors="coerce")

            mean_abs_diff.append(abs(r.mean() - s.mean()))
            std_abs_diff.append(abs(r.std() - s.std()))

        return {
            "mean_abs_diff": float(np.nanmean(mean_abs_diff)),
            "std_abs_diff": float(np.nanmean(std_abs_diff)),
        }


class DistributionSimilarityMetricsEvaluator(Evaluator, ABC):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        numeric_columns, _ = get_schema_columns(ctx)
        if not numeric_columns:
            return {}

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

        if not (ks and wasserstein and t_stats):
            return {}

        return {
            "ks_mean": float(np.nanmean(ks)),
            "wasserstein_mean": float(np.nanmean(wasserstein)),
            "t_stat_mean_abs": float(np.nanmean(t_stats)),
        }


class CategoricalTvMeanEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        _, categorical_columns = get_schema_columns(ctx)
        if not categorical_columns:
            return {}

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
            return {}

        r_corr = ctx.train_df[numeric_columns].corr().fillna(0.0)
        s_corr = ctx.synthetic_df[numeric_columns].corr().fillna(0.0)

        diff = r_corr.values - s_corr.values

        return {
            "corr_fro_diff": float(np.linalg.norm(diff, ord="fro"))
        }
