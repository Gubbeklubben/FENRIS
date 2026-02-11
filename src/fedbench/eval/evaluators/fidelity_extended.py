import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

from .base import Evaluator
from ..context import EvalContext


class ExtendedFidelityEvaluator(Evaluator):
    """
    Optional distributional fidelity metrics.
    Sample-size sensitive — opt-in only.
    """
    name = "fidelity_ext"

    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        train = ctx.train_df
        syn = ctx.synthetic_df
        schema = ctx.schema

        num_cols = [
            c.name for c in schema.columns
            if c.kind in ("continuous", "integer")
        ]

        if not num_cols:
            return {}

        ks_vals = []
        wass_vals = []
        t_vals = []

        for c in num_cols:
            r = pd.to_numeric(train[c], errors="coerce").dropna()
            s = pd.to_numeric(syn[c], errors="coerce").dropna()

            if len(r) == 0 or len(s) == 0:
                continue

            # KS
            ks_stat, _ = stats.ks_2samp(r, s)
            ks_vals.append(ks_stat)

            # Wasserstein
            wass_vals.append(stats.wasserstein_distance(r, s))

            # Two-sample t-statistic (magnitude only)
            t_stat, _ = stats.ttest_ind(r, s, equal_var=False)
            t_vals.append(abs(t_stat))

        return {
            "ks_mean": float(np.mean(ks_vals)) if ks_vals else np.nan,
            "wasserstein_mean": float(np.mean(wass_vals)) if wass_vals else np.nan,
            "t_stat_mean_abs": float(np.mean(t_vals)) if t_vals else np.nan,
        }
