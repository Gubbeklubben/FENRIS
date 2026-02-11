import numpy as np
import pandas as pd
from typing import Dict

from .base import Evaluator
from ..context import EvalContext


class BasicFidelityEvaluator(Evaluator):
    name = "fidelity"

    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        train = ctx.train_df
        syn = ctx.synthetic_df
        schema = ctx.schema

        num_cols = [c.name for c in schema.columns if c.kind in ("continuous", "integer")]
        cat_cols = [c.name for c in schema.columns if c.kind in ("categorical", "binary")]

        out: Dict[str, float] = {}
        out.update(self._numeric_moments(train, syn, num_cols))
        out.update(self._categorical_tv(train, syn, cat_cols))
        out.update(self._corr_fro(train, syn, num_cols))
        return out

    def _numeric_moments(self, r, s, cols):
        if not cols:
            return {"mean_abs_diff": np.nan, "std_abs_diff": np.nan}

        mean_diffs, std_diffs = [], []

        for c in cols:
            r_c = pd.to_numeric(r[c], errors="coerce")
            s_c = pd.to_numeric(s[c], errors="coerce")

            mean_diffs.append(abs(r_c.mean() - s_c.mean()))
            std_diffs.append(abs(r_c.std() - s_c.std()))

        return {
            "mean_abs_diff": float(np.nanmean(mean_diffs)),
            "std_abs_diff": float(np.nanmean(std_diffs)),
        }

    def _categorical_tv(self, r, s, cols):
        if not cols:
            return {"categorical_tv_mean": np.nan}

        tvs = []
        for c in cols:
            pr = r[c].fillna("__NA__").value_counts(normalize=True)
            ps = s[c].fillna("__NA__").value_counts(normalize=True)

            vals = set(pr.index).union(ps.index)
            tv = 0.5 * sum(abs(pr.get(v, 0) - ps.get(v, 0)) for v in vals)
            tvs.append(tv)

        return {"categorical_tv_mean": float(np.mean(tvs))}

    def _corr_fro(self, r, s, cols):
        if len(cols) < 2:
            return {"corr_fro_diff": np.nan}

        r_corr = r[cols].corr().fillna(0.0)
        s_corr = s[cols].corr().fillna(0.0)

        diff = r_corr.values - s_corr.values
        return {"corr_fro_diff": float(np.linalg.norm(diff, ord="fro"))}
