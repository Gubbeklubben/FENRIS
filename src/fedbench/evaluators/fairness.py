from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fedbench.core.eval import EvalContext, Evaluator
from fedbench.util.metrics import fit_tabular_model

_NAN = math.nan

_FAIRNESS_KEYS = (
    "fairness.demographic_parity_diff",
    "fairness.equalized_odds_diff",
    "fairness.equal_opportunity_diff",
)


def _nan_result() -> dict[str, float]:
    return {k: _NAN for k in _FAIRNESS_KEYS}


def _per_group_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = 30,
) -> dict[str, dict[str, int]]:
    """Return per-group TP/FP/TN/FN counts, skipping small groups."""
    out: dict[str, dict[str, int]] = {}
    for g in pd.unique(pd.Series(sensitive)):
        mask = sensitive == g
        if int(mask.sum()) < min_group_size:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        out[str(g)] = {
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "tn": int(((yt == 0) & (yp == 0)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
            "n":  int(mask.sum()),
        }
    return out


def _fairness_metrics_from_counts(
    group_counts: dict[str, dict[str, int]],
) -> dict[str, float]:
    """
    Compute the three fairness metrics from per-group confusion counts.

    Definitions (max – min across groups, NaN groups ignored):
      demographic_parity_diff  = max(pos_rate) – min(pos_rate)
      equal_opportunity_diff   = max(TPR)      – min(TPR)
      equalized_odds_diff      = max(max(ΔTPR, ΔFPR))
    """
    if not group_counts:
        return _nan_result()

    pos_rates, tprs, fprs = [], [], []

    for cm in group_counts.values():
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        n = tp + fp + tn + fn
        if n == 0:
            pos_rates.append(_NAN)
            tprs.append(_NAN)
            fprs.append(_NAN)
            continue
        pos_rates.append((tp + fp) / n)
        tprs.append(tp / (tp + fn) if (tp + fn) else _NAN)
        fprs.append(fp / (fp + tn) if (fp + tn) else _NAN)

    pr = np.array(pos_rates, dtype=float)
    tr = np.array(tprs,      dtype=float)
    fr = np.array(fprs,      dtype=float)

    dp = _NAN if np.all(np.isnan(pr)) else float(np.nanmax(pr) - np.nanmin(pr))

    eopp = _NAN if np.all(np.isnan(tr)) else float(np.nanmax(tr) - np.nanmin(tr))

    if np.all(np.isnan(tr)) or np.all(np.isnan(fr)):
        eo = _NAN
    else:
        delta_tpr = 0.0 if np.all(np.isnan(tr)) else float(np.nanmax(tr) - np.nanmin(tr))
        delta_fpr = 0.0 if np.all(np.isnan(fr)) else float(np.nanmax(fr) - np.nanmin(fr))
        eo = float(max(delta_tpr, delta_fpr))

    return {
        "fairness.demographic_parity_diff": dp,
        "fairness.equalized_odds_diff":     eo,
        "fairness.equal_opportunity_diff":  eopp,
    }


class FairnessEvaluator(Evaluator):
    """
    Evaluates whether synthetic data preserves the fairness properties of real data.

    Strategy
    --------
    1. Train a LogisticRegression classifier on the *synthetic* training data
       (TSTR-style) to predict the task's target column.
    2. Run inference on the *real* training data.
    3. Segment predictions by the sensitive attribute and compute per-group
       TP/FP/TN/FN counts.
    4. Derive the three benchmark-aligned fairness metrics from those counts.

    Prerequisites
    -------------
    - ``ctx.schema`` must expose ``sensitive_column`` and ``target_column``.
    - The task must be binary classification (target encoded as 0/1 or bool).
    - Groups with fewer than ``min_group_size`` samples are excluded from the
      metric computation to avoid unreliable estimates on tiny strata.

    Output keys
    -----------
    - ``fairness.demographic_parity_diff``
    - ``fairness.equalized_odds_diff``
    - ``fairness.equal_opportunity_diff``
    """

    def __init__(self, min_group_size: int = 30, seed: int = 42) -> None:
        self.min_group_size = min_group_size
        self.seed = seed

    def evaluate(self, ctx: EvalContext) -> Mapping[str, float]:
        sensitive_col: str | None = getattr(ctx.schema, "sensitive_column", None)
        target_col:    str | None = getattr(ctx.schema, "target_column",   None)

        if not sensitive_col or not target_col:
            return _nan_result()

        train_df = ctx.train_df
        syn_df   = ctx.synthetic_df

        # Require both columns present in both dataframes
        for col, label in [(sensitive_col, "sensitive"), (target_col, "target")]:
            for df, name in [(train_df, "train"), (syn_df, "synthetic")]:
                if col not in df.columns:
                    return _nan_result()

        feature_cols = [
            c for c in syn_df.columns
            if c not in (sensitive_col, target_col) and c in train_df.columns
        ]
        if not feature_cols:
            return _nan_result()

        X_syn   = syn_df[feature_cols]
        y_syn   = syn_df[target_col]
        X_real  = train_df[feature_cols]
        y_real  = train_df[target_col]
        sens    = train_df[sensitive_col]

        # Ensure binary-numeric target for confusion matrix logic
        try:
            y_syn_enc  = pd.to_numeric(y_syn,  errors="raise").astype(int)
            y_real_enc = pd.to_numeric(y_real, errors="raise").astype(int)
        except (ValueError, TypeError):
            return _nan_result()

        if y_syn_enc.nunique() > 2 or y_real_enc.nunique() > 2:
            # Fairness metrics as defined require binary classification
            return _nan_result()

        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=ctx.seed
        )
        pipe = fit_tabular_model(X_syn, y_syn, model)
        y_pred = pipe.predict(X_real)

        y_true_arr  = y_real_enc.to_numpy()
        y_pred_arr  = np.array(y_pred)
        sensitive_arr = sens.to_numpy()

        group_counts = _per_group_confusion(
            y_true_arr,
            y_pred_arr,
            sensitive_arr,
            min_group_size=self.min_group_size,
        )

        return _fairness_metrics_from_counts(group_counts)