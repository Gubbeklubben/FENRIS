"""Unit tests for the TSTREvaluator (Train on Synthetic, Test on Real).

Approach
--------
TSTR trains a classifier or regressor on synthetic data and evaluates it on
held-out real data.  The evaluator always returns all three keys; inapplicable
paths emit ``float("nan")``:
  * Binary classification  → ``tstr_auc`` computed;
    ``tstr_accuracy`` and ``tstr_rmse`` are nan
  * Multi-class classification → ``tstr_accuracy`` computed;
    ``tstr_auc`` and ``tstr_rmse`` are nan
  * Regression  → ``tstr_rmse`` computed; ``tstr_auc`` and ``tstr_accuracy`` are nan
"""

import math

import numpy as np
import pandas as pd
import pytest

from fenris.builtins.evaluators.utility import TSTREvaluator

from .conftest import make_ctx, make_schema

# ===================================================================
# Guard clauses
# ===================================================================


class TestTSTRGuards:
    """Guard-clause tests: missing prerequisites produce the full nan key set."""

    evaluator = TSTREvaluator()

    EXPECTED_KEYS = {"tstr_auc", "tstr_accuracy", "tstr_rmse"}

    def test_no_target_returns_nan_keys(self):
        """No target_column set → all three keys present as nan."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
        ctx = make_ctx(df, df.copy(), target_column=None)
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_single_class_returns_nan_keys(self):
        """Only one class in the target → all three keys nan."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 0, 0]})
        schema = make_schema(("x", "continuous"), ("target", "binary"))
        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert all(math.isnan(v) for v in result.values())

    def test_returns_all_keys_on_valid_input(self):
        """Key-completeness: all three keys present for a valid binary task."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({"x": rng.normal(0, 1, n), "target": rng.integers(0, 2, n)})
        schema = make_schema(("x", "continuous"), ("target", "binary"))
        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert set(result.keys()) == self.EXPECTED_KEYS


# ===================================================================
# Binary classification path
# ===================================================================


class TestTSTRBinary:
    """Binary: target has exactly two classes → emits ``tstr_auc``."""

    evaluator = TSTREvaluator()

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
            df,
            df.copy(),
            test_df=df.copy(),
            target_column="target",
            schema=schema,
        )
        result = self.evaluator.global_evaluate(ctx)

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
            syn_df,
            syn_df.copy(),
            test_df=test_df,
            target_column="target",
            schema=schema,
        )
        result = self.evaluator.global_evaluate(ctx)

        assert 0.35 < result["tstr_auc"] < 0.65

    def test_binary_emits_tstr_auc_key(self, separable_data):
        """Binary path: tstr_auc computed; tstr_accuracy and tstr_rmse are nan."""
        df, schema = separable_data
        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert "tstr_auc" in result
        assert math.isnan(result["tstr_accuracy"])
        assert math.isnan(result["tstr_rmse"])


# ===================================================================
# Multiclass / categorical classification path
# ===================================================================


class TestTSTRCategorical:
    """Multi-class: target has >2 classes → emits ``tstr_accuracy``."""

    evaluator = TSTREvaluator()

    def test_multiclass_emits_accuracy_key(self):
        """Categorical target → accuracy computed; auc and rmse are nan."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0, 1, n)
        y = rng.choice(["a", "b", "c"], n)
        df = pd.DataFrame({"x": x, "target": y})
        schema = make_schema(("x", "continuous"), ("target", "categorical"))

        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert "tstr_accuracy" in result
        assert math.isnan(result["tstr_auc"])


# ===================================================================
# Regression path
# ===================================================================


class TestTSTRRegression:
    """Regression path: continuous target → emits ``tstr_rmse``."""

    evaluator = TSTREvaluator()

    def test_perfect_linear_gives_low_rmse(self):
        """y = 2*x + 1 → perfectly learnable."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(-10, 10, n)
        y = 2.0 * x + 1.0
        df = pd.DataFrame({"x": x, "target": y})
        schema = make_schema(("x", "continuous"), ("target", "continuous"))

        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert "tstr_rmse" in result
        assert result["tstr_rmse"] < 0.1

    def test_regression_emits_rmse_key(self):
        """Continuous target → rmse computed; auc and accuracy are nan."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "target": rng.normal(0, 1, n),
            }
        )
        schema = make_schema(("x", "continuous"), ("target", "continuous"))

        ctx = make_ctx(df, df.copy(), target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert "tstr_rmse" in result
        assert math.isnan(result["tstr_auc"])
        assert math.isnan(result["tstr_accuracy"])


# ===================================================================
# Label type coercion
# ===================================================================


class TestTSTRLabelCoercion:
    """Regression tests for label type coercion in classification paths.

    Synthetic data often has string-encoded numeric labels (e.g. "0.0", "1.0"
    from a CSV generator) while real data has float labels.  Without explicit
    coercion to str on both sides, sklearn's accuracy_score either raises
    TypeError or silently returns 0.0 (depending on version) because the
    string predictions from the model never equal the float ground-truth labels.

    These tests use perfectly separable data so that a correct implementation
    returns accuracy == 1.0, making a silent wrong result of 0.0 unmistakable.
    Removing the ``y_syn.astype(str)`` / ``y_test.astype(str)`` calls in
    ``TSTREvaluator._compute`` will cause these tests to fail.
    """

    evaluator = TSTREvaluator()

    def _make_separable(self, n_classes: int, rng: np.random.Generator):
        """Return (X, y_str, y_float) with perfectly separated classes.

        Labels in y_str use the same representation that float.astype(str)
        produces ("0.0", "1.0", ...) — matching what the evaluator sees when
        synthetic labels are string-encoded floats.
        """
        rows_per_class = 20
        x_vals = np.concatenate(
            [rng.normal(i * 10, 0.1, rows_per_class) for i in range(n_classes)]
        )
        y_float = pd.Series(
            np.repeat(np.arange(n_classes, dtype=float), rows_per_class)
        )
        y_str = y_float.astype(str)  # "0.0", "1.0", ...
        X = pd.DataFrame({"x": x_vals})
        return X, y_str, y_float

    def test_multiclass_string_labels_in_syn_correct_accuracy(self):
        """Multiclass: syn has string labels, real has float labels.

        With coercion: accuracy == 1.0 (perfect predictor on separable data).
        Without coercion: accuracy == 0.0 (every prediction is wrong because
        str "0.0" != float 0.0) or TypeError on older sklearn.
        """
        rng = np.random.default_rng(0)
        X, y_str, y_float = self._make_separable(n_classes=3, rng=rng)

        syn_df = X.copy()
        syn_df["target"] = y_str  # string labels in synthetic

        real_df = X.copy()
        real_df["target"] = y_float  # float labels in real data

        schema = make_schema(("x", "continuous"), ("target", "categorical"))
        ctx = make_ctx(real_df, syn_df, target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert result["tstr_accuracy"] == pytest.approx(1.0), (
            f"Expected accuracy 1.0 on separable data; got {result['tstr_accuracy']}. "
        )

    def test_binary_string_labels_in_syn_correct_auc(self):
        """Binary: syn has string labels, real has float labels.

        AUC is unaffected by the str/float mismatch (roc_auc_score receives
        float y_test and float y_proba regardless), so this test verifies the
        binary path still produces a valid high AUC — not that it would break
        without coercion, but that coercion does not accidentally break it.
        """
        rng = np.random.default_rng(0)
        X, y_str, y_float = self._make_separable(n_classes=2, rng=rng)

        syn_df = X.copy()
        syn_df["target"] = y_str

        real_df = X.copy()
        real_df["target"] = y_float

        schema = make_schema(("x", "continuous"), ("target", "binary"))
        ctx = make_ctx(real_df, syn_df, target_column="target", schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert result["tstr_auc"] > 0.95, (
            f"Expected high AUC on separable data; got {result['tstr_auc']}."
        )
