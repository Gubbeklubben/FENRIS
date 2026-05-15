"""Cross-cutting edge-case tests applied to all evaluators.

Parameterized to ensure every evaluator handles degenerate inputs gracefully.
Every new evaluator added to ``ALL_EVALUATORS`` is automatically covered by
all three checks without writing any additional test code.

Note on the NaN contract (Code Structure Guide §7.1.2)
------------------------------------------------------
Evaluators emit ``float("nan")`` when a metric is not applicable, rather
than omitting the key.  Tests in this module accept both ``nan`` and finite
values; only ``±inf`` is forbidden.

Note on MIANearestNeighborAttackEvaluator
-----------------------------------------
``global_evaluate`` requires a ``CentralizedEvalContext``.  The edge-case
suite calls ``global_evaluate`` via a ``CentralizedEvalContext`` built by
``make_centralized_ctx`` so the evaluator does not raise a ``TypeError``
before the robustness logic is reached.  All other evaluators receive a
plain ``GlobalEvalContext`` from ``make_ctx``.
"""

import numpy as np
import pandas as pd
import pytest

from fenris.builtins.evaluators.fairness import FairnessEvaluator
from fenris.builtins.evaluators.fidelity import (
    CategoricalTvMeanEvaluator,
    CorrFroDiffEvaluator,
    DistributionSimilarityMetricsEvaluator,
    MomentReductionMetricsEvaluator,
)
from fenris.builtins.evaluators.privacy import (
    AIASupervisedAttackEvaluator,
    DirectOverlapDiagnosticEvaluator,
    MIANearestNeighborAttackEvaluator,
)
from fenris.builtins.evaluators.utility import TSTREvaluator

from .conftest import (
    NUMERIC_DF,
    assert_dicts_nan_safe,
    make_centralized_ctx,
    make_ctx,
)

# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------
# ``_MomentReduction`` and ``_DistSimilarity`` are concrete stub subclasses
# defined in conftest.py (they satisfy the ABC mixin contract so the classes
# can be instantiated).  All nine evaluators are tested in one sweep.
# ---------------------------------------------------------------------------

ALL_EVALUATORS = [
    MomentReductionMetricsEvaluator(),
    DistributionSimilarityMetricsEvaluator(),
    CategoricalTvMeanEvaluator(),
    CorrFroDiffEvaluator(),
    TSTREvaluator(),
    DirectOverlapDiagnosticEvaluator(),
    MIANearestNeighborAttackEvaluator(),
    AIASupervisedAttackEvaluator(),
    FairnessEvaluator(),
]

ALL_EVALUATOR_IDS = [type(e).__name__ for e in ALL_EVALUATORS]


def _make_eval_ctx(evaluator, real, syn, **kwargs):
    """Return the appropriate context type for the given evaluator."""
    if isinstance(evaluator, MIANearestNeighborAttackEvaluator):
        return make_centralized_ctx(real, syn, client_train_df=real, **kwargs)
    return make_ctx(real, syn, **kwargs)


@pytest.mark.parametrize("evaluator", ALL_EVALUATORS, ids=ALL_EVALUATOR_IDS)
class TestEdgeCases:
    def test_single_row_no_crash(self, evaluator):
        """Evaluator must not raise on a single-row DataFrame."""
        real = pd.DataFrame({"x": [1.0], "cat": ["a"]})
        syn = pd.DataFrame({"x": [2.0], "cat": ["b"]})
        ctx = _make_eval_ctx(evaluator, real, syn, target_column="cat")

        # Must not raise — result can be anything (empty dict is fine)
        result = evaluator.global_evaluate(ctx)
        assert isinstance(result, dict)

    def test_deterministic_with_seed(self, evaluator):
        """Same EvalContext → same result on repeated calls.

        Uses ``assert_dicts_nan_safe`` instead of ``dict.__eq__`` to avoid
        relying on CPython's identity-based short-circuit for ``math.nan``.
        """
        ctx = _make_eval_ctx(evaluator, NUMERIC_DF, NUMERIC_DF.copy())
        r1 = evaluator.global_evaluate(ctx)
        r2 = evaluator.global_evaluate(ctx)

        assert_dicts_nan_safe(r1, r2)

    def test_result_values_are_finite_or_nan(self, evaluator):
        """No evaluator should return ±infinity for normal inputs.

        Per the NaN contract (§7.1.2), a missing metric should be
        ``float("nan")`` rather than a missing key.  Both ``nan`` and finite
        values are therefore accepted; only ``±inf`` is forbidden.
        """
        ctx = _make_eval_ctx(evaluator, NUMERIC_DF, NUMERIC_DF * 1.01)
        result = evaluator.global_evaluate(ctx)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
            assert not np.isinf(value), f"{key} = {value} is ±inf (forbidden)"

    def test_all_nan_numeric_column(self, evaluator):
        """Numeric column containing only NaN values must not crash.

        Evaluators that iterate over numeric columns may encounter a column
        where every value is NaN (e.g. after coercion).  The evaluator should
        still return a valid dict with float values.
        """
        real = pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0],
                "bad": [float("nan")] * 3,
            }
        )
        syn = real.copy()
        ctx = _make_eval_ctx(evaluator, real, syn)
        result = evaluator.global_evaluate(ctx)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"

    def test_synthetic_extra_columns_no_crash(self, evaluator):
        """Synthetic data may have columns not in real data — must not crash.

        This can happen when synthetic generators add auxiliary columns.
        Evaluators should silently ignore extra columns.
        """
        real = NUMERIC_DF.copy()
        syn = NUMERIC_DF.copy()
        syn["extra_col"] = 999.0
        ctx = _make_eval_ctx(evaluator, real, syn)
        result = evaluator.global_evaluate(ctx)

        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, float)

    def test_nan_in_synthetic_no_crash(self, evaluator):
        """NaN values in synthetic_df must not cause a crash or ±inf output.

        Real generators (e.g. FedTabDiff) can produce NaN samples.  The
        evaluator must degrade gracefully: return a valid dict of floats
        where each value is either finite or ``nan`` — never ``±inf``.
        """
        real = NUMERIC_DF.copy()
        syn = NUMERIC_DF.copy()
        # Scatter NaN into half the rows of every numeric column
        rng = np.random.default_rng(1)
        nan_mask = rng.random(syn.shape) < 0.5
        syn[nan_mask] = float("nan")

        ctx = _make_eval_ctx(evaluator, real, syn)
        result = evaluator.global_evaluate(ctx)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
            assert not np.isinf(value), f"{key} = {value} is ±inf (forbidden)"

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_inf_in_synthetic_no_crash(self, evaluator):
        """±inf values in synthetic_df must not cause a crash or ±inf output.

        If a generator diverges it can emit ±inf instead of NaN.  The
        evaluator must still return a valid float dict without re-emitting
        the infinity upward.  NumPy may emit a RuntimeWarning when reducing
        over arrays containing ±inf — this is expected and suppressed here.
        """
        real = NUMERIC_DF.copy()
        syn = NUMERIC_DF.copy()
        # Inject +inf and -inf into alternating rows
        syn.iloc[::3, :] = np.inf
        syn.iloc[1::3, :] = -np.inf

        ctx = _make_eval_ctx(evaluator, real, syn)
        result = evaluator.global_evaluate(ctx)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
            assert not np.isinf(value), f"{key} = {value} is ±inf (forbidden)"
