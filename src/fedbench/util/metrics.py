from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fedbench.core.eval import EvalContext

type TaskType = Literal["binary_classification", "multiclass_classification", "regression"]


def make_tabular_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Returns a ColumnTransformer for numeric + categorical preprocessing."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def get_schema_columns(ctx: EvalContext) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) present in train."""
    train_cols = set(ctx.train_df.columns)

    numeric = [
        c.name
        for c in ctx.schema.columns
        if c.kind in ("continuous", "integer") and c.name in train_cols
    ]

    categorical = [
        c.name
        for c in ctx.schema.columns
        if c.kind in ("categorical", "binary") and c.name in train_cols
    ]

    return numeric, categorical


def get_quasi_identifiers(
    all_columns: set[str],
    sensitive_column: str,
    target_column: str | None,
) -> list[str]:
    qi = set(all_columns) - {sensitive_column}
    if target_column:
        qi -= {target_column}
    return sorted(qi)