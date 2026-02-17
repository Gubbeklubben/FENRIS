from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np
from pandas import Series
from scipy import stats

from .base import Evaluator
from ..context import EvalContext


class FidelityEvaluator(Evaluator, ABC):
    pass


class ReductionMetricEvaluator(FidelityEvaluator, ABC):
    def evaluate(self, ctx: EvalContext):
        numeric_columns = [c.name for c in ctx.schema.columns if c.kind in ("continuous", "integer")]
        if not numeric_columns:
            return np.nan

        vals = []

        for col in numeric_columns:
            r = ctx.train_df[col].astype(float)
            s = ctx.test_df[col].astype(float)

            vals.append(self._calculate(r, s))

        return float(np.nanmean(vals))

    @abstractmethod
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        ...


class MeanAbsDiffEvaluator(ReductionMetricEvaluator):
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        return abs(r.mean() - s.mean())


class StdAbsDiffEvaluator(ReductionMetricEvaluator):
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        return abs(r.std() - s.std())


class SampleMetricEvaluator(FidelityEvaluator, ABC):
    def evaluate(self, ctx: EvalContext):
        numeric_columns = [c.name for c in ctx.schema.columns if c.kind in ("continuous", "integer")]
        if not numeric_columns:
            return np.nan

        vals = []

        for col in numeric_columns:
            r = ctx.train_df[col].astype(float).dropna()
            s = ctx.test_df[col].astype(float).dropna()

            if len(r) == 0 or len(s) == 0:
                continue

            vals.append(self._calculate(r, s))

        return float(np.mean(vals)) if vals else np.nan

    @abstractmethod
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        ...


class KsMeanEvaluator(SampleMetricEvaluator):
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        ks_stat, _ = stats.ks_2samp(r, s)
        return ks_stat


class WassersteinMeanEvaluator(SampleMetricEvaluator):
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        return stats.wasserstein_distance(r, s) # type: ignore - method actually accepts 1D arrays in spite of signature


class TStatMeanAbsEvaluator(SampleMetricEvaluator):
    def _calculate(self, r: Series[float], s: Series[float]) -> float:
        t_stat, _ = stats.ttest_ind(r, s, equal_var=False)
        return abs(t_stat)


class CategoricalTvMeanEvaluator(FidelityEvaluator):
    def evaluate(self, ctx: EvalContext):
        categorical_columns = [c.name for c in ctx.schema.columns if c.kind in ("categorical", "binary")]
        if not categorical_columns:
            return np.nan

        tvs = []
        for col in categorical_columns:
            pr = ctx.train_df[col].fillna("__NA__").value_counts(normalize=True)
            ps = ctx.test_df[col].fillna("__NA__").value_counts(normalize=True)

            vals = set(pr.index).union(ps.index)
            tv = 0.5 * sum(abs(pr.get(v, 0) - ps.get(v, 0)) for v in vals)
            tvs.append(tv)

        return float(np.mean(tvs))


class CorrFroDiffEvaluator(FidelityEvaluator):
    def evaluate(self, ctx: EvalContext):
        numeric_columns = [c.name for c in ctx.schema.columns if c.kind in ("continuous", "integer")]
        if len(numeric_columns) < 2:
            return np.nan

        r_corr = ctx.train_df[numeric_columns].corr().fillna(0.0)
        s_corr = ctx.test_df[numeric_columns].corr().fillna(0.0)

        diff = r_corr.values - s_corr.values
        return float(np.linalg.norm(diff, ord="fro"))
