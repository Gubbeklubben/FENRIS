"""Unit tests for ``FairnessEvaluator``.

The evaluator implements a TSTR-style fairness audit:
  1. Trains a LogisticRegression on the **synthetic** data.
  2. Infers on the **real** data, segments predictions by sensitive attribute.
  3. Emits per-column keys:
       ``demographic_parity_diff.{col}``, ``equalized_odds_diff.{col}``,
       ``equal_opportunity_diff.{col}``

Note on the NaN contract (Code Structure Guide §7.1.2)
------------------------------------------------------
Evaluators emit ``float("nan")`` when a metric is not applicable, rather
than omitting the key.  Guard-path tests assert the full nan key set is
returned; metric-path tests assert values are either finite or nan (never ±inf).

Fallback behavior
------------------
If prerequistes fail **before** any sensitive column is processed, the
evaluator returns the generic three-key nan result:
    ``{"demographic_parity_diff": nan, "equalized_odds_diff": nan,
       "equal_opportunity_diff": nan}``

If at least one sensitive column reaches execution (even if it produces nan
values), per-column keys are emitted instead:
    ``{"demographic_parity_diff.{col}": nan, ...}``
"""

import math

import numpy as np
import pandas as pd

from fedbench.builtins.evaluators.fairness import FairnessEvaluator

from .conftest import assert_dicts_nan_safe, make_ctx

# ---------------------------------------------------------------------------
# Sentinel set for quick key-presence assertions (generic fallback)
# ---------------------------------------------------------------------------
GENERIC_NAN_KEYS = frozenset(
    {
        "demographic_parity_diff",
        "equalized_odds_diff",
        "equal_opportunity_diff",
    }
)


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def _biased_df(n_per_group: int = 80) -> pd.DataFrame:
    """Build a dataset where group membership perfectly predicts the label.

    Group 0 (“sensitive” = 0): feature drawn from N(+3, 0.5), target = 1.
    Group 1 (“sensitive” = 1): feature drawn from N(−3, 0.5), target = 0.

    A classifier trained on this data will learn a near-perfect split,
    causing a large ``demographic_parity_diff`` (close to 1.0) because
    group 0 receives almost all positive predictions.

    ``n_per_group`` must be ≥ 30 so that both groups exceed the evaluator’s
    ``min_group_size`` threshold and are included in the metric computation.
    """
    rng = np.random.default_rng(0)
    group0 = pd.DataFrame(
        {
            "feature": rng.normal(3.0, 0.5, n_per_group),
            "sensitive": 0,
            "target": 1,
        }
    )
    group1 = pd.DataFrame(
        {
            "feature": rng.normal(-3.0, 0.5, n_per_group),
            "sensitive": 1,
            "target": 0,
        }
    )
    return pd.concat([group0, group1], ignore_index=True)


# ---------------------------------------------------------------------------
# Guard tests — prerequisites that cause the generic nan fallback
# ---------------------------------------------------------------------------


class TestFairnessGuards:
    """FairnessEvaluator returns the generic nan result for missing prerequisites."""

    evaluator = FairnessEvaluator()

    def test_no_target_column_returns_generic_nan_keys(self):
        """No target_column on the context → generic nan result."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "s": [0, 1, 0]})
        ctx = make_ctx(df, df.copy(), sensitive_columns=("s",))
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == GENERIC_NAN_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_no_sensitive_columns_returns_generic_nan_keys(self):
        """sensitive_columns not set → loop never runs → generic nan result."""
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        ctx = make_ctx(df, df.copy(), target_column="target")
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == GENERIC_NAN_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_empty_sensitive_columns_returns_generic_nan_keys(self):
        """Passing an empty sequence for sensitive_columns → generic nan result."""
        df = pd.DataFrame({"feature": [1.0, 2.0], "target": [0, 1]})
        ctx = make_ctx(df, df.copy(), target_column="target", sensitive_columns=())
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == GENERIC_NAN_KEYS
        assert all(math.isnan(v) for v in result.values())


# ---------------------------------------------------------------------------
# Per-column nan tests — prerequisites fail *after* entering the sensitive loop
# ---------------------------------------------------------------------------


class TestFairnessPerColumnNan:
    """FairnessEvaluator emits per-column nan keys when evaluation fails per-column.

    These tests verify the case where the sensitive-column loop *does* execute
    (so generic fallback is not used) but the per-column computation cannot
    produce a finite result.
    """

    evaluator = FairnessEvaluator()

    def test_non_binary_target_emits_per_column_nan_keys(self):
        """Multi-class target → non-binary check → per-column nan keys emitted."""
        df = pd.DataFrame(
            {
                "feat": [1.0, 2.0, 3.0, 4.0, 5.0],
                "sens": [0, 0, 1, 1, 0],
                "target": [0, 1, 2, 3, 4],
            }
        )
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("sens",),
        )
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["demographic_parity_diff.sens"])
        assert math.isnan(result["equalized_odds_diff.sens"])
        assert math.isnan(result["equal_opportunity_diff.sens"])
        # Must NOT fall back to generic keys (sensitive column was reached)
        assert "demographic_parity_diff" not in result

    def test_insufficient_group_size_emits_per_column_nan_keys(self):
        """Groups smaller than min_group_size (30) → per-column nan keys."""
        small = pd.DataFrame(
            {
                "feat": [1.0, 2.0, 3.0, 4.0],
                "sens": [0, 0, 1, 1],
                "target": [0, 1, 0, 1],
            }
        )
        ctx = make_ctx(
            small,
            small.copy(),
            target_column="target",
            sensitive_columns=("sens",),
        )
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["demographic_parity_diff.sens"])
        assert math.isnan(result["equalized_odds_diff.sens"])
        assert math.isnan(result["equal_opportunity_diff.sens"])

    def test_no_feature_columns_emits_per_column_nan_keys(self):
        """Only target + sensitive in the frame, no features → per-column nan keys."""
        df = pd.DataFrame({"sens": [0, 1, 0, 1], "target": [0, 0, 1, 1]})
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("sens",),
        )
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["demographic_parity_diff.sens"])
        assert math.isnan(result["equalized_odds_diff.sens"])
        assert math.isnan(result["equal_opportunity_diff.sens"])


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


class TestFairnessMetrics:
    """FairnessEvaluator produces meaningful metrics on well-formed data.

    Uses ``_biased_df()`` which has perfectly segregated groups, so the
    trained classifier will predict differently for each group — producing
    measurably large fairness gaps.
    """

    evaluator = FairnessEvaluator()

    def test_output_keys_present_for_sensitive_column(self):
        """Per-column metric keys are emitted when all prerequisites are met."""
        df = _biased_df()
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("sensitive",),
        )
        result = self.evaluator.global_evaluate(ctx)

        assert "demographic_parity_diff.sensitive" in result
        assert "equalized_odds_diff.sensitive" in result
        assert "equal_opportunity_diff.sensitive" in result

    def test_biased_data_yields_large_demographic_parity_diff(self):
        """Strongly segregated groups → demographic_parity_diff > 0.3."""
        df = _biased_df()
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("sensitive",),
        )
        result = self.evaluator.global_evaluate(ctx)

        dp = result["demographic_parity_diff.sensitive"]
        assert math.isfinite(dp), f"expected finite dp_diff, got {dp}"
        assert dp > 0.3, f"expected large bias, got dp_diff={dp:.4f}"

    def test_values_are_finite_or_nan(self):
        """No ±infinity values emitted for valid inputs (NaN contract §7.1.2)."""
        df = _biased_df()
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("sensitive",),
        )
        result = self.evaluator.global_evaluate(ctx)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
            assert not np.isinf(value), f"{key} = {value} is ±inf (forbidden)"

    def test_multiple_sensitive_columns_emit_separate_keys(self):
        """Two sensitive columns → six per-column keys in the result."""
        df = _biased_df()
        # Rename "sensitive" → "group_a", add duplicate as "group_b"
        # Use already-snake_case names to avoid to_snake_case transformations.
        df = df.rename(columns={"sensitive": "group_a"})
        df["group_b"] = df["group_a"]
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("group_a", "group_b"),
        )
        result = self.evaluator.global_evaluate(ctx)

        for col in ("group_a", "group_b"):
            assert f"demographic_parity_diff.{col}" in result
            assert f"equalized_odds_diff.{col}" in result
            assert f"equal_opportunity_diff.{col}" in result

    def test_deterministic_with_same_seed(self):
        """Same EvalContext produces identical results on consecutive calls."""
        df = _biased_df()
        ctx = make_ctx(
            df,
            df.copy(),
            target_column="target",
            sensitive_columns=("sensitive",),
        )
        r1 = self.evaluator.global_evaluate(ctx)
        r2 = self.evaluator.global_evaluate(ctx)

        assert_dicts_nan_safe(r1, r2)
