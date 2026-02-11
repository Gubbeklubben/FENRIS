import numpy as np
import pandas as pd
import pytest

from fedbench.data.schemas import TableSchema, ColumnSchema
from fedbench.eval.context import EvalContext


@pytest.fixture
def schema():
    return TableSchema(columns=(
        ColumnSchema("x", "continuous"),
        ColumnSchema("y", "continuous"),
        ColumnSchema("cat", "categorical"),
        ColumnSchema("label", "binary"),
    ))


@pytest.fixture
def real_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 200),
        "y": rng.normal(5, 2, 200),
        "cat": rng.choice(["a", "b", "c"], 200),
        "label": rng.integers(0, 2, 200),
    })


@pytest.fixture
def split_data(real_data):
    train = real_data.iloc[:150].reset_index(drop=True)
    test = real_data.iloc[150:].reset_index(drop=True)
    return train, test


@pytest.fixture
def synthetic_data(real_data):
    # intentionally similar but not identical
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "x": rng.normal(0.1, 1.1, 300),
        "y": rng.normal(5.2, 2.1, 300),
        "cat": rng.choice(["a", "b", "c"], 300),
        "label": rng.integers(0, 2, 300),
    })


@pytest.fixture
def eval_ctx(schema, split_data, synthetic_data):
    train, test = split_data
    return EvalContext(
        schema=schema,
        train_df=train,
        test_df=test,
        synthetic_df=synthetic_data,
        target_column="label",
        sensitive_columns=["cat"],
        seed=42,
    )
