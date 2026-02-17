from dataclasses import dataclass

import pandas as pd

from fedbench.data.schemas import TableSchema


@dataclass(frozen=True)
class EvalContext:
    schema: TableSchema
    train_df: pd.DataFrame
    seed: int
    test_df: pd.DataFrame | None
    synthetic_df: pd.DataFrame
    target_column: str | None
    sensitive_columns: tuple[str] | None = None