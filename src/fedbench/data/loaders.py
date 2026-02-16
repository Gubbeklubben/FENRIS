from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame

from fedbench.data.schemas import TableSchema, infer_schema


def load_csv(
    file_path: str | Path,
    *,
    header: bool = True,
    custom_encodings: dict[str, Literal["float", "int", "object"]] | None = None,
) -> tuple[DataFrame, TableSchema]:
    """
    Load a CSV file into a DataFrame and return a stabile TableSchema.

    Parameters
    ----------
    file_path
        Path to the CSV file.
    header
        Whether the first row contains column names.  If False, columns
        will be named ``col_0``, ``col_1``, ...
    custom_encodings
        Optional mapping of column name → cast target (``float``, ``int``, ``object``).
        Useful for rare cases where the automatic pandas type inference fails.

    Returns
    -------
    (df, schema)
    """
    # Load
    df = pd.read_csv(str(file_path), header=0 if header else None)

    # Apply custom casts if requested
    if custom_encodings:
        for col, dtype in custom_encodings.items():
            if col in df.columns:
                try:
                    # noinspection PyTypeChecker
                    df[col] = df[col].astype(dtype)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to cast column '{col}' to dtype '{dtype}': {exc}"
                    ) from exc

    # Infer schema (handles both numeric and categorical types)
    schema = infer_schema(df)

    return df, schema
