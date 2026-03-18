import math
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from flwr.common import RecordDict
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fedbench.core.data import TableSchema

type TaskType = Literal[
    "binary_classification", "multiclass_classification", "regression"
]


def count_rdict_bytes(rdict: RecordDict) -> int:
    """
    Count the uncompressed model-parameter payload bytes in a RecordDict.

    Counts only array_records (which carry both real tensors and
    pickle-serialized objects). Excludes metric_records and config_records,
    which carry only scalar values and JSON strings, not model parameters.

    Each Array.data is the raw bytes as stored; shape[0] equals len(data)
    for both numpy arrays (raw buffer) and pickle objects (uint8 encoding).
    """
    total = 0
    for record in rdict.array_records.values():
        for arr in record.values():
            total += len(arr.data)
    return total


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
    numeric_cols: Iterable[str],
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


def safe_nanmean(values: Iterable[float]) -> float:
    """Like ``np.nanmean`` but returns ``nan`` silently for all-NaN inputs.

    ``np.nanmean`` emits a ``RuntimeWarning: Mean of empty slice`` when every
    element is NaN.  This helper avoids that by returning ``math.nan``
    directly when no finite values remain.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    return float(np.mean(finite)) if finite.size else math.nan


def weighted_mean(pairs: Iterable[tuple[float, int]]) -> float:
    """Weighted mean of (value, weight) pairs, NaN-safe."""
    num, den = 0.0, 0
    for v, w in pairs:
        if w > 0 and not math.isnan(v):
            num += float(v) * int(w)
            den += int(w)
    return num / den if den > 0 else math.nan


def weighted_mean_metrics(
    stats: Iterable[tuple[Mapping[str, float], int]],
    keys: Iterable[str],
) -> dict[str, float]:
    """Weighted mean over ``(metrics_dict, weight)`` pairs for each key.

    A multi-key generalization of :func:`weighted_mean`. For each key in
    ``keys``, extracts the per-pair value and computes its row-count-weighted
    mean. Returns ``{key: nan}`` for all keys if ``stats`` is empty.
    """
    keys = list(keys)
    stats = list(stats)
    if not stats:
        return {key: math.nan for key in keys}

    acc: dict[str, list[tuple[float, int]]] = {key: [] for key in keys}

    for metrics, n in stats:
        for key in keys:
            acc[key].append((metrics[key], n))

    return {
        key: weighted_mean(pairs)  # nofmt
        for key, pairs in acc.items()
    }
