from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self

from fedbench.core.eval import Category

type ConfigCls = type[DataConfig] | type[MetricsConfig] | type[Config]


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    schema: str
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
    stop_epsilon: float = 1e-3
    stop_patience: int = 3
    stop_min_rounds: int = 1
    stop_eval_every: int = 1
    stop_synthetic_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.early_stop:
            if self.stop_metric:
                raise ValueError(
                    "early_stop must be enabled when stop_metric is specified"
                )
            return
        if not self.stop_mode:
            raise ValueError(
                "stop_mode must be specified (min or max) when early_stop is enabled"
            )
        if self.stop_epsilon <= 0.0:
            raise ValueError(
                f"stop_epsilon must be a positive float (got {self.stop_epsilon})"
            )
        if self.stop_patience < 1:
            raise ValueError(
                f"stop_patience must be a positive integer (got {self.stop_patience})"
            )
        if self.stop_min_rounds < 1:
            raise ValueError(
                f"stop_min_rounds must be a positive integer "
                f"(got {self.stop_min_rounds})"
            )
        if self.stop_eval_every < 1:
            raise ValueError(
                f"stop_eval_every must be a positive integer "
                f"(got {self.stop_eval_every})"
            )
        if self.stop_synthetic_rows is not None and self.stop_synthetic_rows < 1:
            raise ValueError(
                f"stop_synthetic_rows must be a positive integer or None "
                f"(got {self.stop_synthetic_rows})"
            )


@dataclass(frozen=True)
class SeedConfig:
    """Derived seeds per §23.2 of the technical reference.

    Each randomness source gets a distinct offset so that changing the
    master seed produces a genuinely different experiment.
    """

    master: int
    partitioning: int
    init: int
    training: int
    sampling: int
    evaluation: int

    @classmethod
    def from_master(cls, seed: int = 42) -> SeedConfig:
        return cls(
            master=seed,
            partitioning=seed + 1,
            init=seed + 2,
            training=seed + 3,
            sampling=seed + 4,
            evaluation=seed + 5,
        )


@dataclass(frozen=True)
class Config:
    synthesizer: str
    coordinator: str
    data: DataConfig
    synthesizer_kwargs: dict[
        str,
        None | bool | str | float | int,
    ] = field(default_factory=dict)
    coordinator_kwargs: dict[
        str,
        None | bool | str | float | int,
    ] = field(default_factory=dict)
    num_clients: int = 3
    num_rounds: int = 3
    test_size: float = 0.2
    seed: SeedConfig = field(default_factory=SeedConfig.from_master)
    outputdir: str = ""
    num_synthetic_rows: int | None = None
    client_cpus: float = 2.0
    client_gpus: float = 0.5
    disable_pickle: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self) -> None:
        if self.num_clients < 1:
            raise ValueError(
                f"num_clients must be a positive integer (got {self.num_clients})"
            )
        if self.num_rounds < 1:
            raise ValueError(
                f"num_rounds must be a positive integer (got {self.num_rounds})"
            )
        if self.test_size <= 0.0 or self.test_size >= 1.0:
            raise ValueError(
                f"test_size must be between 0.0 and 1.0 exclusive "
                f"(got {self.test_size})"
            )
        if self.num_synthetic_rows is not None and self.num_synthetic_rows < 1:
            raise ValueError(
                f"num_synthetic_rows must be a positive integer or None "
                f"(got {self.num_synthetic_rows})"
            )

    @classmethod
    def parse_jsons(cls, jsons: str) -> Self:
        cfg = json.loads(jsons)
        data_cfg = cfg.pop("data")
        metrics_cfg = cfg.pop("metrics")
        seed = cfg.pop("seed")
        return cls(
            **cfg,
            data=DataConfig(**data_cfg),
            metrics=MetricsConfig(**metrics_cfg),
            seed=SeedConfig.from_master(seed),
        )

    def jsondict(self) -> dict[str, Any]:
        cfg = asdict(self)
        cfg["seed"] = self.seed.master
        return cfg

    def jsons(self) -> str:
        return json.dumps(self.jsondict())
