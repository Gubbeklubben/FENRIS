from dataclasses import dataclass, field
from pathlib import Path

from flwr_datasets.partitioner import Partitioner


@dataclass(frozen=True)
class Config:
    algorithm_name: str
    dataset: Path
    partitioner_name: str
    partitioner_kwargs: dict[str, Any]
    outputdir: Path
    num_clients: int = 3
    num_rounds: int = 3
    seed: int = 1337
    num_synthetic_rows: int | None = None
    target_col: str | None = None
    sensitive_cols: tuple[str, ...] = field(default_factory=tuple)
