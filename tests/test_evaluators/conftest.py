"""Shared fixtures, helpers, and concrete test doubles for evaluator tests.

Design rationale
----------------
* ``make_ctx`` provides a single construction entry point so that individual
  test modules never need to import or construct ``EvalContext`` directly.
* ``NUMERIC_DF``, ``CATEGORICAL_DF``, and ``MIXED_DF`` are seeded once and
  reused across modules — changes here propagate everywhere automatically.
* ``_MomentReduction`` and ``_DistSimilarity`` live here (not in individual
  test files) because both ``test_fidelity.py`` and ``test_edge_cases.py``
  need them.  They are stub subclasses whose only purpose is to satisfy the
  ABC contract on ``MomentReductionMetricsEvaluator`` and
  ``DistributionSimilarityMetricsEvaluator`` so the classes can be
  instantiated in tests.
"""

import numpy as np
import pandas as pd
import pytest

from fedbench.core.data.schemas import TableSchema, ColumnSchema, infer_schema
from fedbench.core.eval import EvalContext
from fedbench.evaluators.fidelity import (
    DistributionSimilarityMetricsEvaluator,
    MomentReductionMetricsEvaluator,
)


# ---------------------------------------------------------------------------
# EvalContext factory
# ---------------------------------------------------------------------------

def make_ctx(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    test_df: pd.DataFrame | None = None,
    target_column: str | None = None,
    sensitive_columns: tuple[str, ...] | None = None,
    seed: int = 42,
    schema: TableSchema | None = None,
) -> EvalContext:
    """Build an EvalContext with sensible defaults."""
    if test_df is None:
        test_df = train_df.copy()
    if schema is None:
        schema = infer_schema(train_df)
    return EvalContext(
        schema=schema,
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        seed=seed,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
    )


# ---------------------------------------------------------------------------
# Reusable canonical DataFrames (deterministic via seeded RNG)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
N = 200

NUMERIC_DF = pd.DataFrame({
    "age":    _RNG.normal(50, 10, N),
    "income": _RNG.normal(60_000, 15_000, N),
    "score":  _RNG.uniform(0, 1, N),
})

CATEGORICAL_DF = pd.DataFrame({
    "color": _RNG.choice(["red", "blue", "green"], N),
    "size":  _RNG.choice(["S", "M", "L"], N),
})

MIXED_DF = pd.concat([NUMERIC_DF, CATEGORICAL_DF], axis=1)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def make_schema(*col_defs: tuple[str, str]) -> TableSchema:
    """Shorthand: make_schema(("age", "continuous"), ("sex", "binary"))."""
    return TableSchema(tuple(ColumnSchema(n, k) for n, k in col_defs))


# ---------------------------------------------------------------------------
# Concrete stubs for abstract evaluators
# ---------------------------------------------------------------------------
# ``MomentReductionMetricsEvaluator`` and ``DistributionSimilarityMetricsEvaluator``
# both mix in ``ABC``, which prevents direct instantiation.  These empty
# subclasses satisfy the abstract interface without adding any logic of
# their own — they exist purely so pytest can construct evaluator instances.
# ---------------------------------------------------------------------------

class _MomentReduction(MomentReductionMetricsEvaluator):
    """Concrete stub — no overrides; exists only to satisfy the ABC."""


class _DistSimilarity(DistributionSimilarityMetricsEvaluator):
    """Concrete stub — no overrides; exists only to satisfy the ABC."""
