from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from logging import INFO
from typing import TYPE_CHECKING

from flwr.common.logger import log
from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


_BOX_DRAWING = "\u251c\u2500\u2500"


# Quick and dirty
def log_calls(modulename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            log(INFO, f"{modulename}: Calling {func.__name__}")
            log(INFO, f"\t{_BOX_DRAWING} args: {args}")
            log(INFO, f"\t{_BOX_DRAWING} kwargs: {kwargs}")
            ret = func(*args, **kwargs)
            log(INFO, f"\t{_BOX_DRAWING} return value: {ret}\n")
            return ret
        return wrapper
    return decorator


type ModelState = list[NDArray] | dict[str, torch.Tensor]
type ConfigDict = dict[
    str, str | bool | int | float | bytes | list[str] | list[bool] | list[int]
    | list[float] | list[bytes]]
type MetricsDict = dict[str, int | float | list[int] | list[float]]


class MLRuntime(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


@dataclass(frozen=True)
class InitRequest:
    client_id: int
    config: ConfigDict | None


@dataclass(frozen=True)
class InitResponse:
    client_id: int
    statistics: dict[str, NDArray] | None


@dataclass(frozen=True)
class TrainRequest:
    client_id: int
    model_state: ModelState
    config: ConfigDict | None


@dataclass(frozen=True)
class TrainResponse:
    client_id: int
    model_state: ModelState | None
    metrics: MetricsDict | None
    num_examples: int


@dataclass(frozen=True)
class EvalRequest:
    client_id: int
    model_state: ModelState
    config: ConfigDict | None


@dataclass(frozen=True)
class EvalResponse:
    client_id: int
    metrics: MetricsDict | None
