from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from fenris.core.data.schemas import TableSchema


@dataclass(frozen=True)
class EvalContext(ABC):
    """Abstract base for evaluation context objects.

    Attributes
    ----------
    schema : TableSchema
        Schema of the dataset being used.
    synthetic_df : pandas.DataFrame
        Synthetic data produced by the current synthesizer.
    seed : int
        Random seed for reproducible evaluation.
    target_column : str or None
        Name of the target/label column.
    sensitive_columns : tuple[str, ...] or None
        Names of sensitive columns.
    """

    schema: TableSchema
    synthetic_df: pd.DataFrame

    seed: int
    target_column: str | None
    sensitive_columns: tuple[str, ...] | None


@dataclass(frozen=True)
class LocalEvalContext(EvalContext):
    """Evaluation context available on each federated client.

    Extends `EvalContext` with client-local train and test partitions and
    timing information.

    Attributes
    ----------
    train_df : pandas.DataFrame
        Local training partition.
    test_df : pandas.DataFrame
        Local test partition.
    local_train_seconds : float
        Wall-clock seconds spent on local training for this client.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    local_train_seconds: float


@dataclass(frozen=True)
class GlobalEvalContext(EvalContext):
    """Evaluation context available on the server after training.

    Extends `EvalContext` with a holdout set withheld from all clients.

    Attributes
    ----------
    holdout_df : pandas.DataFrame
        Centrally held data that was never distributed to any client.
    """

    holdout_df: pd.DataFrame


@dataclass(frozen=True)
class CentralizedEvalContext(GlobalEvalContext):
    """Extended `GlobalEvalContext` that includes all client training data.

    Use only for evaluators that require access to client training data in
    order to produce meaningful metrics and where no adequate FL-compatible
    alternative exists. Holding all client data centrally is a deliberate
    concession that violates the default federated privacy model.

    Attributes
    ----------
    client_train_df : pandas.DataFrame
        Union of all client training partitions.
    """

    client_train_df: pd.DataFrame
