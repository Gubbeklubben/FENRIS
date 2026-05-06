"""Unit tests for fidelity evaluators.

Approach
--------
Tests use known-answer inputs where the correct result is mathematically
obvious, so no deep statistical expertise is required to verify correctness.
Where the formula allows an exact analytic result (e.g. Wasserstein shift of
1, TV = 0.5, Frobenius = 2√2) the expected value is derived in the docstring.

Note on the NaN contract (Code Structure Guide §7.1.2)
------------------------------------------------------
Evaluators emit ``float("nan")`` for inapplicable metrics rather than
omitting the key.  Tests assert that inapplicable metrics are ``nan`` and
that the full expected key set is always present.
"""

import math

import numpy as np
import pandas as pd
import pytest

from fenris.builtins.evaluators.fidelity import (
    CategoricalTvMeanEvaluator,
    CorrFroDiffEvaluator,
)

from .conftest import (
    CATEGORICAL_DF,
    NUMERIC_DF,
    _DistSimilarity,
    _MomentReduction,
    make_ctx,
    make_schema,
)

# ===================================================================
# MomentReductionMetricsEvaluator
# ===================================================================


class TestMomentReduction:
    """Tests for mean_abs_diff and std_abs_diff metrics."""

    evaluator = _MomentReduction()

    def test_identical_data_gives_zero(self):
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert result["mean_abs_diff"] == pytest.approx(0.0, abs=1e-9)
        assert result["std_abs_diff"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_shift_changes_only_mean(self):
        """Shifting all values by `shift` changes mean_abs_diff but not std."""
        shift = 5.0
        syn = NUMERIC_DF + shift
        ctx = make_ctx(NUMERIC_DF, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert result["mean_abs_diff"] == pytest.approx(shift, abs=1e-6)
        assert result["std_abs_diff"] == pytest.approx(0.0, abs=1e-6)

    def test_scaling_changes_std(self):
        """Doubling values naturally inflates the std of each column."""
        syn = NUMERIC_DF * 2.0
        ctx = make_ctx(NUMERIC_DF, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert result["std_abs_diff"] > 0.0

    def test_single_column(self):
        """Single-column case: mean diff = 3.0, std diff = 0.0 (parallel shift)."""
        real = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
        )
        syn = pd.DataFrame(
            {"x": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
        )
        ctx = make_ctx(real, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert result["mean_abs_diff"] == pytest.approx(3.0, abs=1e-9)
        assert result["std_abs_diff"] == pytest.approx(0.0, abs=1e-9)

    def test_no_numeric_columns_emits_nan_keys(self):
        """No numeric columns → both keys present but nan (NaN contract §7.1.2)."""
        ctx = make_ctx(CATEGORICAL_DF, CATEGORICAL_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["mean_abs_diff"])
        assert math.isnan(result["std_abs_diff"])

    def test_returns_both_keys(self):
        """Sanity-check that both expected keys are present in the output."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert "mean_abs_diff" in result
        assert "std_abs_diff" in result


# ===================================================================
# DistributionSimilarityMetricsEvaluator
# ===================================================================


class TestDistributionSimilarity:
    """Tests for ks_mean, wasserstein_mean, t_stat_mean_abs metrics."""

    evaluator = _DistSimilarity()

    def test_identical_data_gives_zero(self):
        """Perfect replication → all distribution-similarity metrics = 0."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert result["ks_mean"] == pytest.approx(0.0, abs=1e-9)
        assert result["wasserstein_mean"] == pytest.approx(0.0, abs=1e-9)
        assert result["t_stat_mean_abs"] == pytest.approx(0.0, abs=1e-9)

    def test_shifted_wasserstein(self):
        """A uniform shift of 1 gives Wasserstein distance of exactly 1."""
        real = pd.DataFrame({"x": np.arange(100, dtype=float)})
        syn = pd.DataFrame({"x": np.arange(100, dtype=float) + 1.0})
        ctx = make_ctx(real, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert result["wasserstein_mean"] == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_distributions_high_ks(self):
        """Non-overlapping distributions → KS statistic approaches 1."""
        rng = np.random.default_rng(99)
        real = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        syn = pd.DataFrame({"x": rng.normal(100, 1, 500)})
        ctx = make_ctx(real, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert result["ks_mean"] > 0.99

    def test_no_numeric_columns_emits_nan_keys(self):
        """No numeric columns → all keys present but nan (NaN contract §7.1.2)."""
        ctx = make_ctx(CATEGORICAL_DF, CATEGORICAL_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["ks_mean"])
        assert math.isnan(result["wasserstein_mean"])
        assert math.isnan(result["t_stat_mean_abs"])

    def test_returns_all_keys(self):
        """Sanity-check that all three expected keys are present."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF * 1.1)
        result = self.evaluator.global_evaluate(ctx)

        assert "ks_mean" in result
        assert "wasserstein_mean" in result
        assert "t_stat_mean_abs" in result


# ===================================================================
# CategoricalTvMeanEvaluator
# ===================================================================


class TestCategoricalTvMean:
    """Tests for categorical_tv_mean (Total Variation distance)."""

    evaluator = CategoricalTvMeanEvaluator()

    def test_identical_categories_gives_zero(self):
        """Identical category distributions → TV distance = 0."""
        ctx = make_ctx(CATEGORICAL_DF, CATEGORICAL_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert result["categorical_tv_mean"] == pytest.approx(0.0, abs=1e-9)

    def test_completely_disjoint_gives_one(self):
        """All real = 'A', all syn = 'B'  → TV = 1.0."""
        n = 100
        real = pd.DataFrame({"cat": ["A"] * n})
        syn = pd.DataFrame({"cat": ["B"] * n})
        schema = make_schema(("cat", "categorical"))
        ctx = make_ctx(real, syn, schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert result["categorical_tv_mean"] == pytest.approx(1.0, abs=1e-9)

    def test_half_overlap(self):
        """
        real = [A, A, B, B]  → P(A)=0.5, P(B)=0.5
        syn  = [A, A, A, A]  → P(A)=1.0, P(B)=0.0
        TV = 0.5 * (|0.5-1.0| + |0.5-0.0|) = 0.5
        """
        real = pd.DataFrame({"cat": ["A", "A", "B", "B"]})
        syn = pd.DataFrame({"cat": ["A", "A", "A", "A"]})
        schema = make_schema(("cat", "categorical"))
        ctx = make_ctx(real, syn, schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert result["categorical_tv_mean"] == pytest.approx(0.5, abs=1e-9)

    def test_no_categorical_columns_emits_nan_key(self):
        """No categorical columns → key present but nan (NaN contract §7.1.2)."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["categorical_tv_mean"])

    def test_nan_values_treated_as_category(self):
        """NaN is mapped to __NA__ so it shouldn't crash."""
        real = pd.DataFrame({"cat": ["A", None, "B", None]})
        syn = pd.DataFrame({"cat": ["A", None, "B", None]})
        schema = make_schema(("cat", "categorical"))
        ctx = make_ctx(real, syn, schema=schema)
        result = self.evaluator.global_evaluate(ctx)

        assert result["categorical_tv_mean"] == pytest.approx(0.0, abs=1e-9)


# ===================================================================
# CorrFroDiffEvaluator
# ===================================================================


class TestCorrFroDiff:
    """Tests for corr_fro_diff (Frobenius norm of correlation difference)."""

    evaluator = CorrFroDiffEvaluator()

    def test_identical_data_gives_zero(self):
        """Perfect replication → Frobenius norm of the diff matrix = 0."""
        ctx = make_ctx(NUMERIC_DF, NUMERIC_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert result["corr_fro_diff"] == pytest.approx(0.0, abs=1e-9)

    def test_fewer_than_two_columns_emits_nan_key(self):
        """Single numeric column → key present but nan (NaN contract §7.1.2)."""
        real = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        syn = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
        ctx = make_ctx(real, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["corr_fro_diff"])

    def test_perfect_positive_vs_negative_correlation(self):
        """
        real: b = a  → corr = [[1, 1], [1, 1]]
        syn:  b = -a → corr = [[1, -1], [-1, 1]]
        diff = [[0, 2], [2, 0]]
        Frobenius = sqrt(0 + 4 + 4 + 0) = 2*sqrt(2) ≈ 2.828
        """
        a = np.linspace(1, 100, 50)
        real = pd.DataFrame({"a": a, "b": a})
        syn = pd.DataFrame({"a": a, "b": -a})
        ctx = make_ctx(real, syn)
        result = self.evaluator.global_evaluate(ctx)

        expected = 2 * math.sqrt(2)
        assert result["corr_fro_diff"] == pytest.approx(expected, abs=1e-6)

    def test_only_categorical_emits_nan_key(self):
        """No numeric columns → key present but nan (NaN contract §7.1.2)."""
        ctx = make_ctx(CATEGORICAL_DF, CATEGORICAL_DF.copy())
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["corr_fro_diff"])

    def test_constant_column_in_syn_emits_nan_not_zero(self):
        """
        Two numeric columns where one is constant in syn but not in real.

        With the old implementation, safe_corr filtered each DataFrame
        independently.  The constant column survived in the real correlation
        matrix but was dropped from the synthetic one.  The intersection then
        reduced both matrices to 1x1 (identity minus identity = 0), and the
        Frobenius norm was silently returned as 0.0 — a wrong answer, because
        with only one column remaining there is no pairwise correlation
        structure to compare.

        The correct result is NaN: fewer than two non-constant columns survive
        the joint filter, so the metric is undefined.
        """
        a = np.linspace(1, 10, 20)
        real = pd.DataFrame({"a": a, "b": a * 2.0})  # b non-constant in real
        syn = pd.DataFrame({"a": a, "b": [5.0] * 20})  # b constant in syn

        ctx = make_ctx(real, syn)
        result = self.evaluator.global_evaluate(ctx)

        assert math.isnan(result["corr_fro_diff"]), (
            f"Expected NaN when one column is constant in syn, got "
            f"{result['corr_fro_diff']!r}. "
        )
