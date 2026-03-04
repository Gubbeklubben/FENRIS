from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fedbench.core.eval import EvalContext, Evaluator
from fedbench.util.metrics import fit_tabular_model
from fedbench.util.parsing import to_snake_case


def _per_group_confusion(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
        min_group_size: int = 30,
) -> dict[str, dict[str, int]]:
    """Return per-group TP/FP/TN/FN counts, skipping small groups."""
    out: dict[str, dict[str, int]] = {}
    for g in pd.unique(pd.Series(sensitive)):
        mask = (sensitive == g)
        if int(mask.sum()) < min_group_size:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        out[str(g)] = {
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "tn": int(((yt == 0) & (yp == 0)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
            "n": int(mask.sum()),
        }
    return out


def _fairness_metrics_from_counts(
        group_counts: dict[str, dict[str, int]],
) -> tuple[float, float, float]:
    """
    Compute the three fairness metrics from per-group confusion counts.

    Definitions (max – min across groups, NaN groups ignored):
      demographic_parity_diff  = max(pos_rate) – min(pos_rate)
      equal_opportunity_diff   = max(TPR)      – min(TPR)
      equalized_odds_diff      = max(max(ΔTPR, ΔFPR))
    """
    pos_rates, tprs, fprs = [], [], []

    for cm in group_counts.values():
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        n = tp + fp + tn + fn
        if n == 0:
            pos_rates.append(math.nan)
            tprs.append(math.nan)
            fprs.append(math.nan)
            continue

        pos_rate = (tp + fp) / n
        tpr = tp / (tp + fn) if (tp + fn) else math.nan
        fpr = fp / (fp + tn) if (fp + tn) else math.nan

        pos_rates.append(pos_rate)
        tprs.append(tpr)
        fprs.append(fpr)

    pos_rates_a = np.array(pos_rates, dtype=float)
    tprs_a = np.array(tprs, dtype=float)
    fprs_a = np.array(fprs, dtype=float)

    def nanptp(sequence: np.ndarray) -> float:
        """Like np.ptp but ignores NaNs and returns NaN if all values are NaN."""
        if np.all(np.isnan(sequence)):
            return math.nan
        return float(np.nanmax(sequence)) - float(np.nanmin(sequence))

    dp = nanptp(pos_rates_a)
    eopp = nanptp(tprs_a)
    fprs_ptp = nanptp(fprs_a)
    if math.isnan(eopp) or math.isnan(fprs_ptp):
        eo = math.nan
    else:
        eo = float(max(eopp, fprs_ptp))

    return dp, eo, eopp


def _evaluate_for_sensitive_column(
        train_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        target_column: str,
        sensitive_column: str,
        seed: int,
        min_group_size: int = 30,
) -> tuple[float, float, float]:

    nan_result = (math.nan, math.nan, math.nan)

    # Require both columns present in both dataframes
    for col in [sensitive_column, target_column]:
        for df in [train_df, syn_df]:
            if col not in df.columns:
                return nan_result

    feature_columns = [
        col for col in syn_df.columns
        if col not in (sensitive_column, target_column) and col in train_df.columns
    ]
    if not feature_columns:
        return nan_result

    X_syn = syn_df[feature_columns]
    y_syn = syn_df[target_column]
    X_real = train_df[feature_columns]
    y_real = train_df[target_column]
    sens = train_df[sensitive_column]

    # Ensure binary-numeric target for confusion matrix logic
    try:
        y_syn_enc = pd.to_numeric(y_syn, errors="raise").astype(int)
        y_real_enc = pd.to_numeric(y_real, errors="raise").astype(int)
    except (ValueError, TypeError):
        return nan_result

    if np.unique(y_syn_enc).size > 2 or np.unique(y_real_enc).size > 2:
        # Fairness metrics as defined require binary classification
        return nan_result

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )

    try:
        pipe = fit_tabular_model(X_syn, pd.Series(y_syn_enc), model)
    except ValueError:
        return nan_result

    y_pred = pipe.predict(X_real)

    y_true_arr = pd.Series(y_real_enc).to_numpy()
    y_pred_arr = np.array(y_pred)
    sensitive_arr = sens.to_numpy()

    group_counts = _per_group_confusion(
        y_true_arr,
        y_pred_arr,
        sensitive_arr,
        min_group_size=min_group_size,
    )

    if not group_counts:
        return nan_result

    return _fairness_metrics_from_counts(group_counts)


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

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        nan_result = {
            "demographic_parity_diff": math.nan,
            "equalized_odds_diff": math.nan,
            "equal_opportunity_diff": math.nan,
        }

        if not ctx.target_column:
            return nan_result

        metrics: dict[str, float] = {}

        for sensitive_column in ctx.sensitive_columns or []:

            dp, eo, eopp = _evaluate_for_sensitive_column(
                ctx.train_df,
                ctx.synthetic_df,
                target_column=ctx.target_column,
                sensitive_column=sensitive_column,
                seed=ctx.seed,
            )

            sensitive_column_normalized = to_snake_case(sensitive_column)

            metrics[f"demographic_parity_diff.{sensitive_column_normalized}"] = dp
            metrics[f"equalized_odds_diff.{sensitive_column_normalized}"] = eo
            metrics[f"equal_opportunity_diff.{sensitive_column_normalized}"] = eopp

        return metrics or nan_result
