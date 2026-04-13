"""Shared fixtures, helpers, and concrete test doubles for evaluator tests.

Design rationale
----------------
* ``make_ctx`` and ``make_centralized_ctx`` provide the construction entry
  points for ``GlobalEvalContext`` and ``CentralizedEvalContext`` respectively.
  Individual test modules never need to import context classes directly.
* ``NUMERIC_DF``, ``CATEGORICAL_DF``, and ``MIXED_DF`` are seeded once and
  reused across modules — changes here propagate everywhere automatically.
* ``_MomentReduction`` and ``_DistSimilarity`` live here (not in individual
  test files) because both ``test_fidelity.py`` and ``test_edge_cases.py``
  need them.  They are stub subclasses whose only purpose is to satisfy the
  ABC contract on ``MomentReductionMetricsEvaluator`` and
  ``DistributionSimilarityMetricsEvaluator`` so the classes can be
  instantiated in tests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import fedbench.core.data.schemas
from fedbench.builtins.evaluators.fidelity import (
    DistributionSimilarityMetricsEvaluator,
    MomentReductionMetricsEvaluator,
)
from fedbench.core.data.schemas import ColumnSchema, TableSchema, infer_schema
from fedbench.core.eval.evalcontext import (
    CentralizedEvalContext,
    GlobalEvalContext,
    LocalEvalContext,
)

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_dicts_nan_safe(d1: dict[str, float], d2: dict[str, float]) -> None:
    """Assert two metric dicts are equal, treating NaN == NaN.

    Standard ``dict.__eq__`` relies on CPython identity short-circuiting for
    the ``math.nan`` singleton.  This helper is portable: it compares key
    sets explicitly, then checks each value with ``math.isnan`` awareness.
    """
    assert d1.keys() == d2.keys(), f"Key mismatch: {set(d1) ^ set(d2)}"
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if math.isnan(v1):
            assert math.isnan(v2), f"{k}: expected nan, got {v2}"
        else:
            assert v1 == v2, f"{k}: {v1} != {v2}"


# ---------------------------------------------------------------------------
# EvalContext factories
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
) -> GlobalEvalContext:
    """Build a GlobalEvalContext with sensible defaults.

    ``train_df`` is used as the holdout (the real data the server has access
    to).  Pass ``test_df`` explicitly if the caller needs a distinct split;
    it is otherwise ignored for ``GlobalEvalContext`` (which only has
    ``holdout_df``).
    """
    if schema is None:
        schema = infer_schema(train_df)
    return GlobalEvalContext(
        schema=schema,
        holdout_df=train_df,
        synthetic_df=synthetic_df,
        seed=seed,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
    )


def make_local_ctx(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    test_df: pd.DataFrame | None = None,
    target_column: str | None = None,
    sensitive_columns: tuple[str, ...] | None = None,
    seed: int = 42,
    schema: TableSchema | None = None,
) -> LocalEvalContext:
    """Build a LocalEvalContext representing a single federated client."""
    if schema is None:
        schema = infer_schema(train_df)
    if test_df is None:
        test_df = train_df.copy()
    return LocalEvalContext(
        schema=schema,
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        seed=seed,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
        local_train_seconds=math.nan,
    )


def make_centralized_ctx(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    test_df: pd.DataFrame | None = None,
    client_train_df: pd.DataFrame | None = None,
    target_column: str | None = None,
    sensitive_columns: tuple[str, ...] | None = None,
    seed: int = 42,
    schema: TableSchema | None = None,
) -> CentralizedEvalContext:
    """Build a CentralizedEvalContext for evaluators that require client train data.

    Used by ``MIANearestNeighborAttackEvaluator``, which samples members from
    ``client_train_df`` and non-members from ``holdout_df``.  If
    ``client_train_df`` is not provided it defaults to ``train_df``.
    """
    if schema is None:
        schema = infer_schema(train_df)
    if client_train_df is None:
        client_train_df = train_df
    return CentralizedEvalContext(
        schema=schema,
        holdout_df=test_df if test_df is not None else train_df,
        synthetic_df=synthetic_df,
        seed=seed,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
        client_train_df=client_train_df,
    )


# ---------------------------------------------------------------------------
# Reusable canonical DataFrames (deterministic via seeded RNG)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
N = 200

NUMERIC_DF = pd.DataFrame(
    {
        "age": _RNG.normal(50, 10, N),
        "income": _RNG.normal(60_000, 15_000, N),
        "score": _RNG.uniform(0, 1, N),
    }
)

CATEGORICAL_DF = pd.DataFrame(
    {
        "color": _RNG.choice(["red", "blue", "green"], N),
        "size": _RNG.choice(["S", "M", "L"], N),
    }
)

MIXED_DF = pd.concat([NUMERIC_DF, CATEGORICAL_DF], axis=1)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def make_schema(*col_defs: tuple[str, fedbench.core.data.schemas.Kind]) -> TableSchema:
    """Shorthand: make_schema(("age", "continuous"), ("sex", "binary"))."""
    return TableSchema(columns=tuple(ColumnSchema(n, k) for n, k in col_defs))


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
