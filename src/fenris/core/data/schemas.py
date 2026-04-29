from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd

Kind = Literal["continuous", "categorical", "binary", "integer"]

FENRIS_SCHEMA_FORMAT_IDENTIFIER = "FenrisSchema"
FENRIS_SCHEMA_FORMAT_VERSION = "1.0.0"


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
    schema_format: str = FENRIS_SCHEMA_FORMAT_IDENTIFIER
    schema_format_version: str = FENRIS_SCHEMA_FORMAT_VERSION
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

    def numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Return schema columns with kind 'continuous' or 'integer' present in *df*."""
        df_cols = set(df.columns)
        return [
            c.name
            for c in self.columns
            if c.kind in ("continuous", "integer") and c.name in df_cols
        ]

    def nominal_columns(self, df: pd.DataFrame) -> list[str]:
        """Return schema columns with kind 'categorical' or 'binary' present in *df*.

        These are unordered discrete columns for which arithmetic operations
        are not meaningful.
        """
        df_cols = set(df.columns)
        return [
            c.name
            for c in self.columns
            if c.kind in ("categorical", "binary") and c.name in df_cols
        ]


def load_or_infer_schema(schema_path: Path, fallback_df: pd.DataFrame) -> TableSchema:
    if schema_path.exists():
        return _load_schema(schema_path)
    else:
        return infer_schema(fallback_df)


def _load_schema(schema_path: Path) -> TableSchema:
    with schema_path.open() as f:
        root = json.load(f)

    if root["schema_format"] != FENRIS_SCHEMA_FORMAT_IDENTIFIER:
        raise ValueError(f"Schema format {root['schema_format']} is not supported.")
    if root["schema_format_version"] != FENRIS_SCHEMA_FORMAT_VERSION:
        raise ValueError(
            f"FENRIS schema version {root['schema_format_version']} is not supported."
        )

    cols = root.pop("columns")
    return TableSchema(**root, columns=tuple(ColumnSchema(**col) for col in cols))


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
        elif col.nunique(dropna=True) <= 10:  # TODO allow override via CLI
            kind = "categorical"
        elif pd.api.types.is_integer_dtype(dtype):
            kind = "integer"
        elif pd.api.types.is_float_dtype(dtype):
            kind = "continuous"
        else:
            kind = "categorical"

        columns.append(ColumnSchema(name=str(name), kind=kind))

    return TableSchema(columns=tuple(columns))
