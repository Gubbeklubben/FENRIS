from dataclasses import dataclass


@dataclass(frozen=True)
class TrainPlan:
    node_id: int
    model_state: dict[str, np.ndarray]
    config: dict[str, bool | int | float | bytes ]


@dataclass(frozen=True)
class TrainResult:
    model_state: dict[str, np.ndarray]
    metrics: dict[str, float]
    num_examples: int


@dataclass(frozen=True)
class EvalPlan:
    pass


@dataclass(frozen=True)
class EvalResult:
    pass