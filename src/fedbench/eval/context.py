# fed_synth_bench/eval/context.py
from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd

from fedbench.data.schemas import TableSchema


@dataclass(frozen=True)
class EvalContext:
    schema: TableSchema

    # real data
    train_df: pd.DataFrame
    test_df: Optional[pd.DataFrame]

    # synthetic data
    synthetic_df: pd.DataFrame

    # optional metadata
    target_column: Optional[str] = None
    sensitive_columns: Optional[Sequence[str]] = None
    seed: int = 0
