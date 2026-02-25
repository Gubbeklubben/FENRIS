from __future__ import annotations

from abc import ABC
from typing import Mapping

import numpy as np
from scipy import stats

from fedbench.core.eval import EvalContext, Evaluator


class MomentReductionMetricsEvaluator(Evaluator, ABC):
    def evaluate(self, ctx: EvalContext) -> Mapping[str, float]:
        numeric_columns = [c.name for c in ctx.schema.columns if c.kind in ("continuous", "integer")]
        if not numeric_columns:
            return {}

        mean_abs_diff = []
        std_abs_diff = []

        for col in numeric_columns:
            r = ctx.train_df[col].astype(float)
            s = ctx.synthetic_df[col].astype(float)

            mean_abs_diff.append(abs(r.mean() - s.mean()))
            std_abs_diff.append(abs(r.std() - s.std()))

        return {
            "mean_abs_diff": float(np.nanmean(mean_abs_diff)),
            "std_abs_diff": float(np.nanmean(std_abs_diff)),
        }


class DistributionSimilarityMetricsEvaluator(Evaluator, ABC):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        numeric_columns = [c.name for c in ctx.schema.columns if c.kind in ("continuous", "integer")]
        if not numeric_columns:
            return {}

        ks_stats = []
        wasserstein_distances = []
        t_stats_abs = []

        for col in numeric_columns:
            r = ctx.train_df[col].astype(float).dropna()
            s = ctx.synthetic_df[col].astype(float).dropna()

            if len(r) == 0 or len(s) == 0:
                continue

            ks_stat, _ = stats.ks_2samp(r, s)
            ks_stats.append(ks_stat)

            wasserstein_distance = stats.wasserstein_distance(r, s)
            wasserstein_distances.append(wasserstein_distance)

            t_stat, _ = stats.ttest_ind(r, s, equal_var=False)
            t_stats_abs.append(abs(t_stat))

        if not (ks_stats and wasserstein_distances and t_stats_abs):
            return {}

        return {
            "ks_mean": float(np.nanmean(ks_stats)),
            "wasserstein_mean": float(np.nanmean(wasserstein_distances)),
            "t_stat_mean_abs": float(np.nanmean(t_stats_abs)),
        }


class CategoricalTvMeanEvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        categorical_columns = [c.name for c in ctx.schema.columns if c.kind in ("categorical", "binary")]
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
        numeric_columns = [c.name for c in ctx.schema.columns if c.kind in ("continuous", "integer")]
        if len(numeric_columns) < 2:
            return {}

        r_corr = ctx.train_df[numeric_columns].corr().fillna(0.0)
        s_corr = ctx.synthetic_df[numeric_columns].corr().fillna(0.0)

        diff = r_corr.values - s_corr.values

        return {
            "corr_fro_diff": float(np.linalg.norm(diff, ord="fro"))
        }
