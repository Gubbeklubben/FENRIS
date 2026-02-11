import pandas as pd
import numpy as np
from typing import Dict

from .base import Evaluator
from ..context import EvalContext


def _row_hash(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


class PrivacyEvaluator(Evaluator):
    name = "privacy"

    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        train, syn = ctx.train_df, ctx.synthetic_df
        common = list(set(train.columns) & set(syn.columns))
        if not common:
            return {}

        Ht = set(_row_hash(train[common]))
        Hs = _row_hash(syn[common])

        exact_rate = float(np.mean([h in Ht for h in Hs]))

        # identifying columns
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
            Htk = set(_row_hash(train[cols]))
            Hsk = _row_hash(syn[cols])
            partial_rates[k] = float(np.mean([h in Htk for h in Hsk]))

        return {
            "exact_row_match_rate_train": exact_rate,
            "exact_row_match_any": float(exact_rate > 0),
            "partial_match_rate_top1": partial_rates[1],
            "partial_match_rate_top2": partial_rates[2],
            "partial_match_rate_top3": partial_rates[3],
            "partial_match_any": float(any(v > 0 for v in partial_rates.values())),
        }
