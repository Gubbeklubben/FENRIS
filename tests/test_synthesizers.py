"""Tests for synthesizer edge cases.

Regression tests for GitHub issue #35: synthesizers should handle datasets with
only categorical columns or only numerical columns without crashing.

These tests are marked ``torch`` and excluded from the default CI run because
they require PyTorch. Run them explicitly with:

    pytest tests/test_synthesizers.py -m torch

Run a single synthesizer (e.g. fedtabdiff):
    pytest tests/test_synthesizers.py -m torch -k fedtabdiff
"""

import pandas as pd
import pytest

from fenris.app.registry import Group
from fenris.core.algorithm import GlobalInitContext, TrainContext
from fenris.core.data.schemas import ColumnSchema, TableSchema

# Kwargs that reduce compute so the tests stay fast. Synthesizers not listed
# here are instantiated with their defaults (e.g. fed_hello needs no tuning).
_FAST_KWARGS: dict[str, dict] = {
    "fed_simplegan": {"max_batches": 1, "local_epochs": 1},
    "fed_tgan": {"batch_size": 10, "max_batches": 1, "local_epochs": 1},
    "fedtabdiff": {"batch_size": 4, "max_batches": 1, "diffusion_steps": 2},
}


def _synthesizer_names() -> list[str]:
    return list(Group.SYNTHESIZERS.get_registry())


def _make_instance(name: str):
    factory = Group.SYNTHESIZERS.get_registry().load(name)
    return factory(**_FAST_KWARGS.get(name, {}))


def _run_global_init_and_train(synth, df: pd.DataFrame, schema: TableSchema) -> None:
    init_ctx = GlobalInitContext(coordinator="fake_coordinator", seed=0, schema=schema)
    artifacts = synth.global_init(df.copy(), init_ctx)

    train_ctx = TrainContext(
        coordinator="fake_coordinator",
        seed=1,
        schema=schema,
        global_init_artifacts=artifacts.synthesizer,
        client_storage=None,
    )
    synth.train(artifacts.coordinator, df.copy(), train_ctx)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cat_only_schema() -> TableSchema:
    return TableSchema(
        columns=(
            ColumnSchema("sex", "binary"),
            ColumnSchema("status", "categorical"),
        )
    )


@pytest.fixture
def num_only_schema() -> TableSchema:
    return TableSchema(
        columns=(
            ColumnSchema("age", "integer"),
            ColumnSchema("weight", "continuous"),
        )
    )


@pytest.fixture
def mixed_schema() -> TableSchema:
    return TableSchema(
        columns=(
            ColumnSchema("age", "integer"),
            ColumnSchema("weight", "continuous"),
            ColumnSchema("sex", "binary"),
            ColumnSchema("status", "categorical"),
        )
    )


@pytest.fixture
def cat_only_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sex": ["M", "F", "M", "F", "M", "M", "F", "M", "F", "M"],
            "status": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
        }
    )


@pytest.fixture
def num_only_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 60, 35, 40, 55, 28, 33],
            "weight": [70.0, 80.0, 65.0, 90.0, 75.0, 85.0, 60.0, 95.0, 72.0, 68.0],
        }
    )


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 60, 35, 40, 55, 28, 33],
            "weight": [70.0, 80.0, 65.0, 90.0, 75.0, 85.0, 60.0, 95.0, 72.0, 68.0],
            "sex": ["M", "F", "M", "F", "M", "M", "F", "M", "F", "M"],
            "status": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
        }
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.torch
@pytest.mark.parametrize("synthesizer_name", _synthesizer_names())
def test_cat_only_does_not_raise(
    synthesizer_name, cat_only_df, cat_only_schema
) -> None:
    """Synthesizers should work correctly with a dataset that has only categorical
    columns. Regression test for issue #35.
    """
    _run_global_init_and_train(
        _make_instance(synthesizer_name), cat_only_df, cat_only_schema
    )


@pytest.mark.torch
@pytest.mark.parametrize("synthesizer_name", _synthesizer_names())
def test_num_only_does_not_raise(
    synthesizer_name, num_only_df, num_only_schema
) -> None:
    """Synthesizers should work correctly with a dataset that has only numerical
    columns. Regression test for issue #35.
    """
    _run_global_init_and_train(
        _make_instance(synthesizer_name), num_only_df, num_only_schema
    )


@pytest.mark.torch
@pytest.mark.parametrize("synthesizer_name", _synthesizer_names())
def test_mixed_does_not_raise(synthesizer_name, mixed_df, mixed_schema) -> None:
    """Synthesizers should work correctly with a dataset that has both numerical and
    categorical columns.
    """
    _run_global_init_and_train(_make_instance(synthesizer_name), mixed_df, mixed_schema)
