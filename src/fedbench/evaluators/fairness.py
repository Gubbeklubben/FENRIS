"""
Fairness evaluators.

Measures whether the synthetic data preserves the fairness properties of
the real training data. Fairness is assessed via a TSTR-style binary
classifier whose predictions are stratified by a protected (sensitive)
attribute. Reports demographic parity, equal opportunity, and equalized
odds differences across groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fedbench.core.eval import Evaluator, LocalEvalContext
from fedbench.core.eval.evalcontext import GlobalEvalContext
from fedbench.core.logger import log_debug
from fedbench.util.metrics import fit_tabular_model
from fedbench.util.parsing import to_snake_case


@dataclass
class _PerGroupConfusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


class _FairnessMetrics(NamedTuple):
    demographic_parity_diff: float = math.nan
    equalized_odds_diff: float = math.nan
    equal_opportunity_diff: float = math.nan


class FairnessEvaluator(Evaluator):
    """Evaluate whether synthetic data preserves the fairness properties of real data.

    A TSTR-style :class:`~sklearn.linear_model.LogisticRegression` classifier is
    trained on *synthetic* data, then evaluated on *real* training data.
    Predictions are segmented by the sensitive attribute to derive per-group
    confusion matrices and fairness metrics.

    Notes
    -----
    Requires ``ctx.sensitive_columns`` and ``ctx.target_column`` to be set.
    The task must be binary classification (target encoded as 0/1 or bool).
    Groups with fewer than ``min_group_size`` samples are excluded from metric
    computation to avoid unreliable estimates on tiny strata.

    Reports the following output metrics per sensitive column:

    * ``fairness.demographic_parity_diff.<column>``
    * ``fairness.equalized_odds_diff.<column>``
    * ``fairness.equal_opportunity_diff.<column>``
    """

    MIN_GROUP_SIZE = 30

    # noinspection PyMethodMayBeStatic
    def _per_group_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
        min_group_size: int = MIN_GROUP_SIZE,
    ) -> dict[str, _PerGroupConfusion]:
        """Return per-group TP/FP/TN/FN counts, skipping small groups."""
        out: dict[str, _PerGroupConfusion] = {}
        for g in pd.unique(pd.Series(sensitive)):
            mask = sensitive == g
            if int(mask.sum()) < min_group_size:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            out[str(g)] = _PerGroupConfusion(
                tp=int(((yt == 1) & (yp == 1)).sum()),
                fp=int(((yt == 0) & (yp == 1)).sum()),
                tn=int(((yt == 0) & (yp == 0)).sum()),
                fn=int(((yt == 1) & (yp == 0)).sum()),
            )
        return out

    # noinspection PyMethodMayBeStatic
    def _nanptp(self, sequence: np.ndarray) -> float:
        """NaN-safe peak-to-peak (max − min). Returns NaN if all values are NaN."""
        if np.all(np.isnan(sequence)):
            return math.nan
        return float(np.nanmax(sequence)) - float(np.nanmin(sequence))

    def _fairness_metrics_from_counts(
        self,
        group_counts: Mapping[str, _PerGroupConfusion],
    ) -> _FairnessMetrics:
        """Compute the three fairness metrics from per-group confusion counts.

        Parameters
        ----------
        group_counts :
            Mapping from group label to a ``PerGroupConfusion`` instance with
            fields ``tp``, ``fp``, ``tn``, ``fn``.

        Returns
        -------
        _FairnessMetrics
            Named tuple with fields ``demographic_parity_diff``,
            ``equalized_odds_diff``, and ``equal_opportunity_diff``.
            Returns all-NaN ``FairnessMetrics`` if ``group_counts`` is empty.

        Notes
        -----
        Metrics are defined as max – min across groups (NaN groups ignored):

        * ``demographic_parity_diff`` = ptp(positive_rate)
        * ``equal_opportunity_diff``  = ptp(TPR)
        * ``equalized_odds_diff``     = max(ptp(TPR), ptp(FPR))
        """
        if not group_counts:
            return _FairnessMetrics()

        pos_rates, tprs, fprs = [], [], []

        for cm in group_counts.values():
            tp, fp, tn, fn = cm.tp, cm.fp, cm.tn, cm.fn
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

        dp = self._nanptp(pos_rates_a)
        eopp = self._nanptp(tprs_a)
        fprs_ptp = self._nanptp(fprs_a)
        if math.isnan(eopp) or math.isnan(fprs_ptp):
            eo = math.nan
        else:
            eo = float(max(eopp, fprs_ptp))

        return _FairnessMetrics(
            demographic_parity_diff=dp,
            equalized_odds_diff=eo,
            equal_opportunity_diff=eopp,
        )

    def _get_group_counts_for_column(
        self,
        train_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        target_column: str,
        sensitive_column: str,
        seed: int,
        min_group_size: int = MIN_GROUP_SIZE,
    ) -> dict[str, _PerGroupConfusion]:
        """Run TSTR prediction and return raw per-group confusion matrix counts."""
        for col in [sensitive_column, target_column]:
            for df in [train_df, syn_df]:
                if col not in df.columns:
                    log_debug(
                        "Fairness",
                        f"Column '{col}' missing from DataFrame "
                        f"with columns {df.columns.tolist()}",
                    )
                    return {}

        feature_columns = [
            col
            for col in syn_df.columns
            if col not in (sensitive_column, target_column) and col in train_df.columns
        ]
        if not feature_columns:
            log_debug(
                "Fairness",
                f"No usable feature columns found. "
                f"syn_df columns: {syn_df.columns.tolist()}, "
                f"train_df columns: {train_df.columns.tolist()}",
            )
            return {}

        X_syn = syn_df[feature_columns]
        y_syn = syn_df[target_column]
        X_real = train_df[feature_columns]
        y_real = train_df[target_column]
        sens = train_df[sensitive_column]

        try:
            y_syn_enc = pd.to_numeric(y_syn, errors="raise").astype(int)
            y_real_enc = pd.to_numeric(y_real, errors="raise").astype(int)
        except (ValueError, TypeError):
            log_debug(
                "Fairness",
                "Target column contains non-numeric values "
                "that cannot be encoded as integers.",
            )
            return {}

        if np.unique(y_syn_enc).size > 2 or np.unique(y_real_enc).size > 2:
            log_debug(
                "Fairness", "Target column must be binary (exactly two unique values)."
            )
            return {}

        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
        try:
            pipe = fit_tabular_model(X_syn, pd.Series(y_syn_enc), model)
        except ValueError as e:
            log_debug(
                "Fairness",
                "Error fitting model. "
                "Check that feature columns are valid and contain no NaNs.",
            )
            log_debug("Fairness", str(e))
            return {}

        y_pred = pipe.predict(X_real)
        return self._per_group_confusion(
            pd.Series(y_real_enc).to_numpy(),
            np.array(y_pred),
            sens.to_numpy(),
            min_group_size=min_group_size,
        )

    # noinspection PyMethodMayBeStatic
    def _nan_result(self) -> dict[str, float]:
        return {
            "demographic_parity_diff": math.nan,
            "equalized_odds_diff": math.nan,
            "equal_opportunity_diff": math.nan,
        }

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        if not ctx.target_column:
            return self._nan_result()

        metrics: dict[str, float] = {}

        for sensitive_column in ctx.sensitive_columns or []:
            group_counts = self._get_group_counts_for_column(
                train_df=ctx.holdout_df,
                syn_df=ctx.synthetic_df,
                target_column=ctx.target_column,
                sensitive_column=sensitive_column,
                seed=ctx.seed,
            )

            result = self._fairness_metrics_from_counts(group_counts)
            key = to_snake_case(sensitive_column)

            metrics[f"demographic_parity_diff.{key}"] = result.demographic_parity_diff
            metrics[f"equalized_odds_diff.{key}"] = result.equalized_odds_diff
            metrics[f"equal_opportunity_diff.{key}"] = result.equal_opportunity_diff

        return metrics or self._nan_result()

    def local_evaluate(
        self, ctx: LocalEvalContext
    ) -> dict[str, dict[str, _PerGroupConfusion]]:
        """Train on syn_df, predict on real train_df, and report per-group confusion
        matrix cells for each sensitive column.

        Only integer cell counts (TP, FP, TN, FN) are sent — no raw data.
        Exact federated aggregation: cells are additive integers.
        """
        if not ctx.target_column:
            return {}

        col_counts: dict[str, dict[str, _PerGroupConfusion]] = {}

        for sensitive_column in ctx.sensitive_columns or []:
            col_counts[sensitive_column] = self._get_group_counts_for_column(
                train_df=ctx.train_df,
                syn_df=ctx.synthetic_df,
                target_column=ctx.target_column,
                sensitive_column=sensitive_column,
                seed=ctx.seed,
                min_group_size=self.MIN_GROUP_SIZE,
            )

        return col_counts

    def aggregate(
        self,
        stats: Iterable[Mapping[str, Mapping[str, _PerGroupConfusion]]],
    ) -> dict[str, float]:
        """Sum confusion matrix cells across clients, then derive fairness metrics.

        Exact aggregation: each real record belongs to exactly one client, so
        the global confusion matrix equals the element-wise sum of local matrices.
        The ``MIN_GROUP_SIZE`` guard is re-applied on aggregated totals before
        emitting metrics to avoid unreliable estimates from small groups.
        """
        agg: dict[str, dict[str, _PerGroupConfusion]] = {}

        for st in stats:
            for sensitive_column, group_counts in st.items():
                agg.setdefault(sensitive_column, {})
                for group, cm in group_counts.items():
                    acc = agg[sensitive_column].setdefault(group, _PerGroupConfusion())
                    acc.tp += cm.tp
                    acc.fp += cm.fp
                    acc.tn += cm.tn
                    acc.fn += cm.fn

        metrics: dict[str, float] = {}

        for sensitive_column, group_counts in agg.items():
            valid_counts = {
                g: cm
                for g, cm in group_counts.items()
                if cm.tp + cm.fp + cm.tn + cm.fn >= self.MIN_GROUP_SIZE
            }

            result = self._fairness_metrics_from_counts(valid_counts)
            key = to_snake_case(sensitive_column)

            metrics[f"demographic_parity_diff.{key}"] = result.demographic_parity_diff
            metrics[f"equalized_odds_diff.{key}"] = result.equalized_odds_diff
            metrics[f"equal_opportunity_diff.{key}"] = result.equal_opportunity_diff

        return metrics or self._nan_result()
