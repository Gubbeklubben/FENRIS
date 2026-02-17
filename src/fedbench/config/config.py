import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Self


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    partitioner: str
    partitioner_kwargs: dict[
        str, None | bool | str | float | int
    ] = field(default_factory=dict)
    target_col: str | None = None
    sensitive_cols: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MetricsConfig:
    run_categories: tuple[str, ...] = field(default_factory=tuple)
    early_stop: bool = False
    stop_metric: str | None = None
    stop_mode: Literal["min", "max"] | None = None
    stop_epsilon: float | None = None
    stop_patience: int | None = None
    stop_min_rounds: int | None = None
    stop_eval_every: int | None = None
    stop_synthetic_rows: int | None = None


@dataclass(frozen=True)
class Config:
    algorithm: str
    num_clients: int
    num_rounds: int
    test_size: float
    seed: int
    outputdir: str
    data: DataConfig
    num_synthetic_rows: int | None = None
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def parse_jsons(cls, jsons: str) -> Self:
        cfg = json.loads(jsons)
        data_cfg = cfg.pop("data")
        metrics_cfg = cfg.pop("metrics")
        return cls(
            **cfg,
            data=DataConfig(**data_cfg),
            metrics=MetricsConfig(**metrics_cfg)
        )

    def jsons(self) -> str:
        return json.dumps(asdict(self))
