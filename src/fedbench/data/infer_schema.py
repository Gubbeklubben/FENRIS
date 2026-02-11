from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from .schemas import TableSchema, ColumnSchema, Kind


def infer_schema(df: DataFrame) -> TableSchema:
    """Create a `TableSchema` from a pandas DataFrame.

    Rules:
    - float / np.float* → "continuous"
    - int / np.integer* → "integer"
    - cardinality <= 2 → "binary"
    - anything else → "categorical"
    """
    columns: list[ColumnSchema] = []

    for name, col in df.items():
        dtype = col.dtype

        # default for non-numeric columns with cardinality > 2
        kind: Kind = "categorical"

        # 1️⃣ numeric → continuous / integer
        if pd.api.types.is_float_dtype(dtype):
            kind = "continuous"
        elif col.nunique(dropna=True) <= 2:
            kind = "binary"
        elif pd.api.types.is_integer_dtype(dtype):
            kind = "integer"

        columns.append(ColumnSchema(name=str(name), kind=kind))

    return TableSchema(tuple(columns))
