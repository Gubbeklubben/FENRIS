from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from .schemas import TableSchema, ColumnSchema

def infer_schema(df: DataFrame) -> TableSchema:
    """Create a `TableSchema` from a pandas DataFrame.

    Rules:
    - float / np.float* → "continuous"
    - int / np.integer* → "integer"
    - object / np.object* → "categorical"
          (based on unique value count < 10 → binary, < 32 → categorical)
    """
    columns: list[ColumnSchema] = []

    for name, col in df.items():
        dtype = col.dtype

        # 1️⃣ numeric → continuous / integer
        if pd.api.types.is_float_dtype(dtype):
            kind = "continuous"
        elif pd.api.types.is_integer_dtype(dtype):
            # binary columns are also integers but with two values
            uniqs = col.nunique(dropna=True)
            kind = "binary" if uniqs <= 2 else "integer"
        elif pd.api.types.is_object_dtype(dtype):
            uniqs = col.nunique(dropna=True)
            # small cardinality → categorical / binary
            if uniqs <= 2:
                kind = "binary"
            elif uniqs <= 32:  # arbitrary threshold for “categorical”
                kind = "categorical"
            else:
                # fallback, treat strings as categorical anyway
                kind = "categorical"
        else:
            # for any other dtype we default to categorical (worst‑case)
            kind = "categorical"

        columns.append(ColumnSchema(name=name, kind=kind))

    return TableSchema(tuple(columns))
