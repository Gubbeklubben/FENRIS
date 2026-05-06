from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd

Kind = Literal["continuous", "categorical", "binary", "integer"]

FENRIS_SCHEMA_FORMAT_IDENTIFIER = "FenrisSchema"
FENRIS_SCHEMA_FORMAT_VERSION = "1.0.0"


@dataclass(frozen=True)
class ColumnSchema:
    """Schema for a single table column.

    Attributes
    ----------
    name : str
        Column name.
    kind : Kind
        Semantic type: ``"continuous"``, ``"categorical"``, ``"binary"``, or
        ``"integer"``.
    """

    name: str
    kind: Kind


@dataclass(frozen=True)
class TableSchema:
    """Schema for a tabular dataset.

    Attributes
    ----------
    schema_format : str
        Format identifier; always ``"FenrisSchema"``.
    schema_format_version : str
        Schema format version string.
    columns : tuple[ColumnSchema, ...]
        Ordered tuple of column schemas.
    """

    schema_format: str = FENRIS_SCHEMA_FORMAT_IDENTIFIER
    schema_format_version: str = FENRIS_SCHEMA_FORMAT_VERSION
    columns: Tuple[ColumnSchema, ...] = field(default_factory=tuple)

    def lookup(self, name: str) -> ColumnSchema:
        """Return the `ColumnSchema` for *name*.

        Parameters
        ----------
        name : str
            Column name to look up.

        Returns
        -------
        ColumnSchema

        Raises
        ------
        KeyError
            If *name* is not present in the schema.
        """
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Column '{name}' not found in schema.")

    def kind_of(self, name: str) -> str:
        """Return the `Kind` of column *name* as a string.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        str
        """
        return self.lookup(name).kind

    def numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Return schema columns with kind ``"continuous"`` or ``"integer"`` present in
        *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame used to filter columns to those actually present.

        Returns
        -------
        list[str]
        """
        df_cols = set(df.columns)
        return [
            c.name
            for c in self.columns
            if c.kind in ("continuous", "integer") and c.name in df_cols
        ]

    def nominal_columns(self, df: pd.DataFrame) -> list[str]:
        """Return schema columns with kind ``"categorical"`` or ``"binary"`` present in
        *df*.

        These are unordered discrete columns for which arithmetic operations
        are not meaningful.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame used to filter columns to those actually present.

        Returns
        -------
        list[str]
        """
        df_cols = set(df.columns)
        return [
            c.name
            for c in self.columns
            if c.kind in ("categorical", "binary") and c.name in df_cols
        ]


def load_or_infer_schema(schema_path: Path, fallback_df: pd.DataFrame) -> TableSchema:
    """Load a schema from *schema_path*, or infer one from *fallback_df*.

    Parameters
    ----------
    schema_path : Path
        Path to a JSON schema file. Loaded if it exists; ignored otherwise.
    fallback_df : pandas.DataFrame
        DataFrame used to infer a schema when *schema_path* does not exist.

    Returns
    -------
    TableSchema
    """
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

    Column kinds are inferred using the following rules, evaluated in order:

    - cardinality ≤ 2 → ``"binary"``
    - cardinality ≤ 10 → ``"categorical"``
    - integer dtype → ``"integer"``
    - float dtype → ``"continuous"``
    - any other type → ``"categorical"``

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to infer the schema from.

    Returns
    -------
    TableSchema
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
