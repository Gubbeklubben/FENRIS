from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from logging import DEBUG, INFO
from typing import TYPE_CHECKING

from flwr.common.logger import log as _flwr_log
from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


_BOX_DRAWING = "\u251c\u2500\u2500"


def log(header: str, message_lines: tuple[str, ...], level=INFO):
    _flwr_log(level, header)
    for line in message_lines:
        _flwr_log(level, f"\t{_BOX_DRAWING} {line}")


# Quick and dirty, set and export env variable FLWR_LOG_LEVEL="DEBUG" to enable.
def log_calls(modulename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _flwr_log(DEBUG, f"{modulename}: Calling {func.__name__}")
            _flwr_log(DEBUG, f"\t{_BOX_DRAWING} args: {args}")
            _flwr_log(DEBUG, f"\t{_BOX_DRAWING} kwargs: {kwargs}")
            ret = func(*args, **kwargs)
            _flwr_log(DEBUG, f"\t{_BOX_DRAWING} return value: {ret}")
            _flwr_log(DEBUG, "")
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

    def create_response(self, statistics: dict[str, NDArray]) -> InitResponse:
        return InitResponse(self.client_id, statistics)


@dataclass(frozen=True)
class InitResponse:
    client_id: int
    statistics: dict[str, NDArray] | None


@dataclass(frozen=True)
class TrainRequest:
    client_id: int
    model_state: ModelState
    config: ConfigDict | None

    def create_response(
            self,
            model_state: ModelState | None,
            metrics: MetricsDict | None,
            num_examples: int) -> TrainResponse:

        return TrainResponse(self.client_id, model_state, metrics, num_examples)


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

    def create_response(self, metrics: MetricsDict | None) -> EvalResponse:
        return EvalResponse(self.client_id, metrics)


@dataclass(frozen=True)
class EvalResponse:
    client_id: int
    metrics: MetricsDict | None
