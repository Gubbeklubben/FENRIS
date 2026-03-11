import hashlib
import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fedbench.core.data import TableSchema

type TaskType = Literal[
    "binary_classification", "multiclass_classification", "regression"
]

NAN_TOKEN = "__NaN__"


def make_tabular_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Returns a ColumnTransformer for numeric + categorical preprocessing."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # fmt: off
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    )),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    # fmt: on

    return preprocessor


def fit_tabular_model(X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> Pipeline:
    preprocessor = make_tabular_preprocessor(X)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X, y)
    return pipe


def get_numeric_columns(df: pd.DataFrame, schema: TableSchema) -> list[str]:
    """Return schema columns with kind 'continuous' or 'integer' present in df."""
    df_cols = set(df.columns)
    return [
        c.name
        for c in schema.columns
        if c.kind in ("continuous", "integer") and c.name in df_cols
    ]


def get_nominal_columns(df: pd.DataFrame, schema: TableSchema) -> list[str]:
    """Return schema columns with kind 'categorical' or 'binary' present in df.

    These are unordered discrete columns for which arithmetic operations
    are not meaningful.
    """
    df_cols = set(df.columns)
    return [
        c.name
        for c in schema.columns
        if c.kind in ("categorical", "binary") and c.name in df_cols
    ]


def sanitize_numeric_df(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """
    Return a numeric-only dataframe safe for statistics.

    Steps
    -----
    1. Select numeric columns
    2. Coerce non-numeric values to NaN
    3. Replace ±inf with NaN
    4. Drop rows containing NaN

    Result
    ------
    DataFrame containing only finite numeric values.
    Safe for numpy/scipy/sklearn operations.
    """
    clean: pd.DataFrame = (
        df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return clean


def safe_nanmean(values: list[float]) -> float:
    """Like ``np.nanmean`` but returns ``nan`` silently for all-NaN inputs.

    ``np.nanmean`` emits a ``RuntimeWarning: Mean of empty slice`` when every
    element is NaN.  This helper avoids that by returning ``math.nan``
    directly when no finite values remain.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    return float(np.mean(finite)) if finite.size else math.nan


def get_quasi_identifiers(
    all_columns: set[str],
    sensitive_column: str,
    target_column: str | None,
) -> list[str]:
    qi = set(all_columns) - {sensitive_column}
    if target_column:
        qi -= {target_column}
    return sorted(qi)


def canonical_value(val: Any) -> str:
    """Deterministic string representation for a value."""
    if pd.isna(val):
        return NAN_TOKEN
    # explicit bool handling before int (bool is a subclass of int)
    if isinstance(val, (bool, np.bool_)):
        return str(int(val))
    if isinstance(val, (np.floating, float)):
        return f"{float(val):.8f}"
    if isinstance(val, (np.integer, int)):
        return str(int(val))
    if isinstance(val, str):
        return val.strip()
    return str(val)


def canonical_row_hash(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df = df[df.columns.sort_values()]

    def hash_row(row: pd.Series) -> str:
        canonical = [canonical_value(v) for v in row.values]
        joined = "|".join(canonical)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    return df.apply(hash_row, axis=1)
