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

from .conftest import NUMERIC_DF, make_ctx, make_schema


# ===================================================================
# DirectOverlapDiagnosticEvaluator
# ===================================================================

class TestDirectOverlap:
    """Tests for exact-match and partial-match memorisation diagnostics."""

    evaluator = DirectOverlapDiagnosticEvaluator()

    EXPECTED_KEYS = {
        "exact_row_match_rate_train",
        "exact_row_match_any",
        "partial_match_rate_top1",
        "partial_match_rate_top2",
        "partial_match_rate_top3",
        "partial_match_any",
    }

    def test_full_memorization(self):
        """syn == train  → 100 % exact match."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.evaluate(ctx)

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
        ctx = make_ctx(NUMERIC_DF, syn)
        result = self.evaluator.evaluate(ctx)

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
        ctx = make_ctx(NUMERIC_DF, syn)
        result = self.evaluator.evaluate(ctx)

        assert 0.4 < result["exact_row_match_rate_train"] < 0.6

    def test_disjoint_columns_emits_nan_keys(self):
        """No shared columns → all 6 keys present as nan (NaN contract §7.1.2)."""
        real = pd.DataFrame({"a": [1, 2]})
        syn = pd.DataFrame({"b": [3, 4]})
        ctx = make_ctx(real, syn)
        result = self.evaluator.evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_returns_all_keys(self):
        """Key-completeness check: all six expected metric keys must be present."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS


# ===================================================================
# MIANearestNeighborAttackEvaluator
# ===================================================================

class TestMIA:
    """Tests for Membership Inference Attack using nearest-neighbour distance.

    MIA checks whether an attacker can distinguish training members from
    non-members by their distance to the synthetic data.  If syn ≈ train,
    members are closer → AUC > 0.5; if syn is random, AUC ≈ 0.5.
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

        ctx = make_ctx(train, syn, test_df=test)
        result = self.evaluator.evaluate(ctx)

        assert result["mia_auc"] > 0.7

    def test_random_syn_near_chance(self):
        """syn unrelated to both train and test → AUC near 0.5."""
        rng = np.random.default_rng(0)
        n = 200
        train = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        test = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        # Synthetic from a completely different region
        syn = pd.DataFrame({"x": rng.normal(50, 1, n), "y": rng.normal(50, 1, n)})

        ctx = make_ctx(train, syn, test_df=test)
        result = self.evaluator.evaluate(ctx)

        # Both members and non-members are far from syn → no signal
        assert 0.35 < result["mia_auc"] < 0.65

    def test_empty_train_returns_nan_keys(self):
        """Empty training set → all three MIA keys present as nan (NaN contract §7.1.2)."""
        empty = pd.DataFrame({"x": pd.Series(dtype=float)})
        syn = pd.DataFrame({"x": [1.0, 2.0]})
        ctx = make_ctx(empty, syn, test_df=empty)
        result = self.evaluator.evaluate(ctx)

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

        ctx = make_ctx(train, syn, test_df=test)
        result = self.evaluator.evaluate(ctx)

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
        result = self.evaluator.evaluate(ctx)

        assert set(result.keys()) == self.GENERIC_NAN_KEYS
        assert all(math.isnan(v) for v in result.values())

        ctx2 = make_ctx(df, df.copy(), sensitive_columns=())
        result2 = self.evaluator.evaluate(ctx2)
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
            test_df=df.copy(),
            sensitive_columns=("sensitive",),
            schema=schema,
        )
        result = self.evaluator.evaluate(ctx)

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
            test_df=df.copy(),
            sensitive_columns=("sens_val",),
            schema=schema,
        )
        result = self.evaluator.evaluate(ctx)

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
        result = self.evaluator.evaluate(ctx)

        assert math.isnan(result["aia_accuracy.sensitive"])
        assert math.isnan(result["aia_auc.sensitive"])
        assert math.isnan(result["aia_rmse.sensitive"])
