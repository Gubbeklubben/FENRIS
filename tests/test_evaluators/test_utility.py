"""Unit tests for the TSTREvaluator (Train on Synthetic, Test on Real).

Approach
--------
TSTR trains a classifier or regressor on synthetic data and evaluates it on
held-out real data.  Tests exercise three paths:
  * Binary classification  → emits ``tstr_auc`` and ``tstr_accuracy``
  * Multi-class classification → emits ``tstr_accuracy`` (no AUC)
  * Regression  → emits ``tstr_rmse``

Known bug
---------
``TSTREvaluator`` crashes on single-class data (sklearn LogisticRegression
requires ≥2 distinct classes).  The corresponding edge-case test in
``test_edge_cases.py`` is marked ``xfail`` until a guard is added.
"""

import numpy as np
import pandas as pd
import pytest

from fedbench.evaluators.utility import TSTREvaluator

from .conftest import make_ctx, make_schema


evaluator = TSTREvaluator()


# ===================================================================
# Guard clauses
# ===================================================================

class TestTSTRGuards:
    """Guard-clause tests: evaluator should return {} when it cannot run."""

    def test_no_target_returns_empty(self):
        """No target_column set → TSTR has nothing to predict; returns {}."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
        ctx = make_ctx(df, df.copy(), target_column=None)
        assert evaluator.evaluate(ctx) == {}


# ===================================================================
# Binary classification path
# ===================================================================

class TestTSTRBinary:

    @pytest.fixture()
    def separable_data(self):
        """Linearly separable binary task: y=1 when x > 0."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = (x > 0).astype(int)
        df = pd.DataFrame({"x": x, "target": y})
        schema = make_schema(("x", "continuous"), ("target", "binary"))
        return df, schema

    def test_separable_gives_high_auc(self, separable_data):
        """Synthetic data identical to real → AUC should exceed 0.85."""
        df, schema = separable_data
        ctx = make_ctx(
            df, df.copy(),
            test_df=df.copy(),
            target_column="target",
            schema=schema,
        )
        result = evaluator.evaluate(ctx)

        assert "tstr_auc" in result
        assert result["tstr_auc"] > 0.85

    def test_random_labels_give_chance_auc(self):
        """Random target → AUC near 0.5."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.normal(0, 1, n)
        syn_df = pd.DataFrame({"x": x, "target": rng.integers(0, 2, n)})
        test_df = pd.DataFrame({"x": x, "target": rng.integers(0, 2, n)})
        schema = make_schema(("x", "continuous"), ("target", "binary"))

        ctx = make_ctx(
            syn_df, syn_df.copy(),
            test_df=test_df,
            target_column="target",
            schema=schema,
        )
        result = evaluator.evaluate(ctx)

        assert 0.2 < result["tstr_auc"] < 0.8

    def test_binary_emits_tstr_auc_key(self, separable_data):
        df, schema = separable_data
        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = evaluator.evaluate(ctx)

        assert "tstr_auc" in result
        assert "tstr_accuracy" not in result
        assert "tstr_rmse" not in result


# ===================================================================
# Multiclass / categorical classification path
# ===================================================================

class TestTSTRCategorical:

    def test_multiclass_emits_accuracy_key(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0, 1, n)
        y = rng.choice(["a", "b", "c"], n)
        df = pd.DataFrame({"x": x, "target": y})
        schema = make_schema(("x", "continuous"), ("target", "categorical"))

        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = evaluator.evaluate(ctx)

        assert "tstr_accuracy" in result
        assert "tstr_auc" not in result


# ===================================================================
# Regression path
# ===================================================================

class TestTSTRRegression:

    def test_perfect_linear_gives_low_rmse(self):
        """y = 2*x + 1 → perfectly learnable."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(-10, 10, n)
        y = 2.0 * x + 1.0
        df = pd.DataFrame({"x": x, "target": y})
        schema = make_schema(("x", "continuous"), ("target", "continuous"))

        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = evaluator.evaluate(ctx)

        assert "tstr_rmse" in result
        assert result["tstr_rmse"] < 0.1

    def test_regression_emits_rmse_key(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "target": rng.normal(0, 1, n),
        })
        schema = make_schema(("x", "continuous"), ("target", "continuous"))

        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = evaluator.evaluate(ctx)

        assert "tstr_rmse" in result
        assert "tstr_auc" not in result
        assert "tstr_accuracy" not in result
