from abc import ABC
from dataclasses import dataclass

import pandas as pd

from fenris.core.data.schemas import TableSchema


@dataclass(frozen=True)
class EvalContext(ABC):
    schema: TableSchema
    synthetic_df: pd.DataFrame

    seed: int
    target_column: str | None
    sensitive_columns: tuple[str, ...] | None


@dataclass(frozen=True)
class LocalEvalContext(EvalContext):
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    local_train_seconds: float


@dataclass(frozen=True)
class GlobalEvalContext(EvalContext):
    holdout_df: pd.DataFrame


@dataclass(frozen=True)
class CentralizedEvalContext(GlobalEvalContext):
    """
    Extended GlobalEvalContext for evaluators that have a specific need to
    violate the default FL privacy model (in which client data should stay on clients).
    This is a pragmatic concession to enable centralized evaluation
    where there exists no adequate substitute for real client train data.
    """

    client_train_df: pd.DataFrame
