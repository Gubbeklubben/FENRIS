from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Self

from fedbench.core.eval import Category

type ConfigCls = type[DataConfig] | type[MetricsConfig] | type[Config]


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    partitioner: str
    partitioner_kwargs: dict[str, None | bool | str | float | int] = field(
        default_factory=dict
    )
    target_col: str | None = None
    sensitive_cols: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MetricsConfig:
    run_categories: tuple[Category, ...] = field(default_factory=tuple)
    early_stop: bool = False
    stop_metric: str | None = None
    stop_mode: Literal["min", "max"] | None = None
    stop_epsilon: float | None = None
    stop_patience: int | None = None
    stop_min_rounds: int | None = None
    stop_eval_every: int | None = None
    stop_synthetic_rows: int | None = None


@dataclass(frozen=True)
class SeedConfig:
    """Derived seeds per §23.2 of the technical reference.

    Each randomness source gets a distinct offset so that changing the
    master seed produces a genuinely different experiment.
    """

    partitioning: int  # s + 1
    init: int  # s + 2
    sampling: int  # s + 3
    evaluation: int  # s + 4

    @classmethod
    def from_master(cls, seed: int = 42) -> SeedConfig:
        return cls(
            partitioning=seed + 1,
            init=seed + 2,
            sampling=seed + 3,
            evaluation=seed + 4,
        )


@dataclass(frozen=True)
class Config:
    algorithm: str
    data: DataConfig
    algorithm_kwargs: dict[
        str,
        None | bool | str | float | int,
    ] = field(default_factory=dict)
    num_clients: int = 3
    num_rounds: int = 3
    test_size: float = 0.2
    seed: SeedConfig = field(default_factory=SeedConfig.from_master)
    outputdir: str = ""
    num_synthetic_rows: int | None = None
    disable_pickle: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self) -> None:
        if self.num_clients < 1:
            raise ValueError(f"Number of clients {self.num_clients} is not supported")
        if self.num_rounds < 1:
            raise ValueError(f"Number of rounds {self.num_rounds} is not supported")
        if self.test_size <= 0.0 or self.test_size >= 1.0:
            raise ValueError(
                f"Test size {self.test_size} is not supported, must be between 0 and 1"
            )
        if self.num_synthetic_rows is not None and self.num_synthetic_rows < 1:
            raise ValueError(
                f"Number of synthetic rows {self.num_synthetic_rows} is not supported"
            )

    @classmethod
    def parse_jsons(cls, jsons: str) -> Self:
        cfg = json.loads(jsons)
        data_cfg = cfg.pop("data")
        metrics_cfg = cfg.pop("metrics")
        seed_cfg = cfg.pop("seed")
        return cls(
            **cfg,
            data=DataConfig(**data_cfg),
            metrics=MetricsConfig(**metrics_cfg),
            seed=SeedConfig(**seed_cfg),
        )

    def jsons(self) -> str:
        return json.dumps(asdict(self))
