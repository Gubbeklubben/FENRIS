"""Shared helpers for evaluator implementations.

These functions are used by multiple evaluator modules (fidelity, utility,
privacy, fairness, scalability) but are not part of the public evaluator API.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def fit_tabular_model(x: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> Pipeline:
    preprocessor = make_tabular_preprocessor(x)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(x, y)
    return pipe


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
