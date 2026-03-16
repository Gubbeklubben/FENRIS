"""Unit tests for privacy evaluators: DirectOverlap, MIA, AIA.

Approach
--------
Each evaluator is tested by constructing datasets where the expected
outcome is obvious from first principles (e.g. copying training data
produces 100 % overlap; fully disjoint data produces 0 %).

Note on the NaN contract (Code Structure Guide §7.1.2)
------------------------------------------------------
Evaluators emit ``float("nan")`` for inapplicable metrics rather than
omitting the key.  Tests assert the full key set is present and values are nan.
"""

import math
import numpy as np
import pandas as pd
import pytest

from fedbench.evaluators.privacy import (
    AIASupervisedAttackEvaluator,
    DirectOverlapDiagnosticEvaluator,
    MIANearestNeighborAttackEvaluator,
)

from .conftest import NUMERIC_DF, make_ctx, make_centralized_ctx, make_local_ctx, make_schema


# ===================================================================
# DirectOverlapDiagnosticEvaluator
# ===================================================================

class TestDirectOverlap:
    """Tests for exact-match and partial-match memorization diagnostics.

    ``global_evaluate`` intentionally returns NaN for this evaluator —
    overlap against a server holdout is a structural false negative (§16.5).
    All meaningful tests use ``local_evaluate`` + ``aggregate``, which is the
    only correct mode for overlap detection.
    """

    evaluator = DirectOverlapDiagnosticEvaluator()

    EXPECTED_KEYS = {
        "exact_row_match_rate_train",
        "exact_row_match_any",
        "partial_match_rate_top1",
        "partial_match_rate_top2",
        "partial_match_rate_top3",
        "partial_match_any",
    }

    def test_global_evaluate_returns_nan(self):
        """global_evaluate is unsupported by design — must return all-NaN result."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_full_memorization(self):
        """syn == train → 100 % exact match via local_evaluate + aggregate."""
        ctx = make_local_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.aggregate([self.evaluator.local_evaluate(ctx)])

        assert result["exact_row_match_rate_train"] == pytest.approx(1.0)
        assert result["exact_row_match_any"] == 1.0

    def test_no_overlap(self):
        """Completely different synthetic data → 0 % match."""
        rng = np.random.default_rng(999)
        syn = pd.DataFrame({
            "age":    rng.normal(200, 1, len(NUMERIC_DF)),
            "income": rng.normal(999_999, 1, len(NUMERIC_DF)),
            "score":  rng.uniform(100, 200, len(NUMERIC_DF)),
        })
        ctx = make_local_ctx(NUMERIC_DF, syn)
        result = self.evaluator.aggregate([self.evaluator.local_evaluate(ctx)])

        assert result["exact_row_match_rate_train"] == pytest.approx(0.0)
        assert result["exact_row_match_any"] == 0.0

    def test_partial_overlap(self):
        """First half from train, second half random → ~50 % match."""
        half = len(NUMERIC_DF) // 2
        rng = np.random.default_rng(123)
        random_half = pd.DataFrame({
            "age":    rng.normal(200, 1, half),
            "income": rng.normal(999_999, 1, half),
            "score":  rng.uniform(100, 200, half),
        })
        syn = pd.concat(
            [NUMERIC_DF.iloc[:half].reset_index(drop=True), random_half],
            ignore_index=True,
        )
        ctx = make_local_ctx(NUMERIC_DF, syn)
        result = self.evaluator.aggregate([self.evaluator.local_evaluate(ctx)])

        assert 0.4 < result["exact_row_match_rate_train"] < 0.6

    def test_disjoint_columns_emits_nan_keys(self):
        """No shared columns → local_evaluate returns None → aggregate emits all-NaN."""
        real = pd.DataFrame({"a": [1, 2]})
        syn = pd.DataFrame({"b": [3, 4]})
        ctx = make_local_ctx(real, syn)
        result = self.evaluator.aggregate([self.evaluator.local_evaluate(ctx)])

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_returns_all_keys(self):
        """Key-completeness check: all six expected metric keys must be present."""
        ctx = make_local_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.aggregate([self.evaluator.local_evaluate(ctx)])

        assert set(result.keys()) == self.EXPECTED_KEYS


# ===================================================================
# MIANearestNeighborAttackEvaluator
# ===================================================================

class TestMIA:
    """Tests for Membership Inference Attack using nearest-neighbor distance.

    MIA checks whether an attacker can distinguish training members from
    non-members by their distance to the synthetic data.  If syn ≈ train,
    members are closer → AUC > 0.5; if syn is random, AUC ≈ 0.5.

    ``global_evaluate`` requires a ``CentralizedEvalContext``: members are
    sampled from ``client_train_df`` and non-members from ``holdout_df``.
    """

    evaluator = MIANearestNeighborAttackEvaluator()

    EXPECTED_KEYS = {"mia_auc", "mia_accuracy", "mia_advantage"}

    def _make_disjoint_datasets(self, rng, n=200):
        """Create train and test with different distributions."""
        train = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        test = pd.DataFrame({
            "x": rng.normal(10, 1, n),
            "y": rng.normal(10, 1, n),
        })
        return train, test

    def test_memorized_syn_high_auc(self):
        """syn == train → members are close, non-members far → high AUC."""
        rng = np.random.default_rng(0)
        train, test = self._make_disjoint_datasets(rng, n=200)
        syn = train.copy()

        ctx = make_centralized_ctx(train, syn, test_df=test, client_train_df=train)
        result = self.evaluator.global_evaluate(ctx)

        assert result["mia_auc"] > 0.7

    def test_random_syn_near_chance(self):
        """syn unrelated to both train and test → AUC near 0.5."""
        rng = np.random.default_rng(0)
        n = 200
        train = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        test = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        # Synthetic from a completely different region
        syn = pd.DataFrame({"x": rng.normal(50, 1, n), "y": rng.normal(50, 1, n)})

        ctx = make_centralized_ctx(train, syn, test_df=test, client_train_df=train)
        result = self.evaluator.global_evaluate(ctx)

        # Both members and non-members are far from syn → no signal
        assert 0.35 < result["mia_auc"] < 0.65

    def test_empty_train_returns_nan_keys(self):
        """Empty training set → all three MIA keys present as nan (NaN contract §7.1.2)."""
        empty = pd.DataFrame({"x": pd.Series(dtype=float)})
        syn = pd.DataFrame({"x": [1.0, 2.0]})
        ctx = make_centralized_ctx(empty, syn, test_df=empty, client_train_df=empty)
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["mia_auc"])
        assert math.isnan(result["mia_accuracy"])
        assert math.isnan(result["mia_advantage"])

    def test_returns_all_keys(self):
        """Key-completeness check: all three MIA metric keys must be present."""
        rng = np.random.default_rng(42)
        n = 200
        train = pd.DataFrame({"x": rng.normal(0, 1, n)})
        test = pd.DataFrame({"x": rng.normal(5, 1, n)})
        syn = train.copy()

        ctx = make_centralized_ctx(train, syn, test_df=test, client_train_df=train)
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS


# ===================================================================
# AIASupervisedAttackEvaluator
# ===================================================================

class TestAIA:
    """Tests for Attribute Inference Attack (supervised regression/classification).

    AIA trains a model to infer a sensitive attribute from quasi-identifier
    columns in the synthetic data, then measures accuracy on real test data.
    """

    evaluator = AIASupervisedAttackEvaluator()

    GENERIC_NAN_KEYS = {"aia_accuracy", "aia_auc", "aia_rmse"}

    def test_no_sensitive_columns_returns_nan_keys(self):
        """No sensitive_columns → generic nan result emitted (NaN contract §7.1.2)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
        ctx = make_ctx(df, df.copy(), sensitive_columns=None)
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == self.GENERIC_NAN_KEYS
        assert all(math.isnan(v) for v in result.values())

        ctx2 = make_ctx(df, df.copy(), sensitive_columns=())
        result2 = self.evaluator.global_evaluate(ctx2)
        assert set(result2.keys()) == self.GENERIC_NAN_KEYS
        assert all(math.isnan(v) for v in result2.values())

    def test_learnable_binary_sensitive_attr(self):
        """sensitive = (x > 0) → perfectly learnable from x."""
        rng = np.random.default_rng(0)
        n = 300
        x = rng.normal(0, 3, n)
        sensitive = (x > 0).astype(int)
        df = pd.DataFrame({"x": x, "sensitive": sensitive})
        schema = make_schema(("x", "continuous"), ("sensitive", "binary"))

        ctx = make_ctx(
            df, df.copy(),
            sensitive_columns=("sensitive",),
            schema=schema,
        )
        result = self.evaluator.global_evaluate(ctx)

        assert "aia_accuracy.sensitive" in result
        assert result["aia_accuracy.sensitive"] > 0.8

    def test_regression_sensitive_emits_rmse(self):
        """Continuous sensitive column → aia_rmse key."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0, 1, n)
        s = 2.0 * x + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "sens_val": s})
        schema = make_schema(("x", "continuous"), ("sens_val", "continuous"))

        ctx = make_ctx(
            df, df.copy(),
            sensitive_columns=("sens_val",),
            schema=schema,
        )
        result = self.evaluator.global_evaluate(ctx)

        assert "aia_rmse.sens_val" in result

    def test_no_quasi_identifiers_emits_nan_keys(self):
        """Only sensitive + target columns, no QIs → per-column keys present as nan."""
        df = pd.DataFrame({"target": [0, 1, 0, 1], "sensitive": [1, 0, 1, 0]})
        schema = make_schema(("target", "binary"), ("sensitive", "binary"))

        ctx = make_ctx(
            df, df.copy(),
            target_column="target",
            sensitive_columns=("sensitive",),
            schema=schema,
        )
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["aia_accuracy.sensitive"])
        assert math.isnan(result["aia_auc.sensitive"])
        assert math.isnan(result["aia_rmse.sensitive"])


# ===================================================================
# AIA label type coercion
# ===================================================================


class TestAIALabelCoercion:
    """Regression tests for label type coercion in the AIA categorical path.

    The AIA evaluator calls accuracy_score(y_test, y_pred) where y_pred is
    produced by a model trained on synthetic labels.  If synthetic labels are
    string-encoded ("0.0", "1.0") but real test labels are floats (0.0, 1.0),
    accuracy_score either raises TypeError or silently returns 0.0 depending
    on the sklearn version.

    These tests use perfectly separable data so accuracy == 1.0 is expected,
    making a wrong result of 0.0 unmistakable.  Removing the
    ``y_syn.astype(str)`` / ``y_test.astype(str)`` calls in
    ``AIASupervisedAttackEvaluator._compute_column`` will cause these tests
    to fail.
    """

    evaluator = AIASupervisedAttackEvaluator()

    def _make_separable_aia(self, rng: np.random.Generator, n_per_class: int = 30):
        """Return (syn_df, real_df, schema) for a perfectly separable AIA task.

        Sensitive attribute is binary: 0 when x < 0, 1 when x > 0.
        syn_df uses string-encoded labels; real_df uses float labels.
        """
        x_neg = rng.normal(-5, 0.1, n_per_class)
        x_pos = rng.normal( 5, 0.1, n_per_class)
        x = np.concatenate([x_neg, x_pos])

        y_float = pd.Series(
            [0.0] * n_per_class + [1.0] * n_per_class
        )
        y_str = y_float.astype(str)  # "0.0", "1.0"

        syn_df = pd.DataFrame({"x": x, "sensitive": y_str})
        real_df = pd.DataFrame({"x": x, "sensitive": y_float})
        schema = make_schema(("x", "continuous"), ("sensitive", "binary"))
        return syn_df, real_df, schema

    def test_categorical_string_labels_correct_accuracy(self):
        """AIA categorical: syn has string labels, real has float labels.

        With coercion: accuracy == 1.0 on separable data.
        Without coercion: accuracy == 0.0 or TypeError.
        """
        rng = np.random.default_rng(0)
        syn_df, real_df, schema = self._make_separable_aia(rng)

        from .conftest import make_ctx
        ctx = make_ctx(
            real_df, syn_df,
            sensitive_columns=("sensitive",),
            schema=schema,
        )
        result = self.evaluator.global_evaluate(ctx)

        assert result["aia_accuracy.sensitive"] == pytest.approx(1.0), (
            f"Expected accuracy 1.0 on separable data; "
            f"got {result['aia_accuracy.sensitive']}. "
        )
