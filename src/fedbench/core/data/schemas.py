from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import pandas as pd

Kind = Literal["continuous", "categorical", "binary", "integer"]


# --------------------------------------------------------------------------- #
# Column → keeps the column name (string) and a broad Semantic *kind*
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnSchema:
    name: str
    kind: Kind


# --------------------------------------------------------------------------- #
# Table keeps an ordered tuple of ColumnSchema objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableSchema:
    columns: Tuple[ColumnSchema, ...] = field(default_factory=tuple)

    # convenience: look‑up helper
    def lookup(self, name: str) -> ColumnSchema:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Column '{name}' not found in schema.")

    # handy short‑cut for checking column kinds
    def kind_of(self, name: str) -> str:
        return self.lookup(name).kind


def infer_schema(df: pd.DataFrame) -> TableSchema:
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

        kind: Kind

        if col.nunique(dropna=True) <= 2:
            kind = "binary"
        elif (
            col.nunique(dropna=True) <= 10
        ):  # magic number TODO get from config, implement ability to override from CLI
            kind = "categorical"
        elif pd.api.types.is_integer_dtype(dtype):
            kind = "integer"
        elif pd.api.types.is_float_dtype(dtype):
            kind = "continuous"
        else:
            kind = "categorical"

        # uncomment print statement for debug
        # print(f"{name} ({kind})")

        columns.append(ColumnSchema(name=str(name), kind=kind))

    return TableSchema(tuple(columns))
