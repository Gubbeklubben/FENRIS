from dataclasses import dataclass
import pandas as pd

from fedbench.data.schemas import TableSchema


@dataclass(frozen=True)
class EvalContext:
    schema: TableSchema

    # real data
    train_df: pd.DataFrame
    test_df: pd.DataFrame | None

    # synthetic data
    synthetic_df: pd.DataFrame

    seed: int
    target_column: str | None
    sensitive_columns: tuple[str, ...] | None = None