from __future__ import annotations

from dataclasses import dataclass, field
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any

from flwr.common.logger import log as _flwr_log
from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type Arrays  = list[NDArray] | dict[str, torch.Tensor]
type Objects = dict[str, Any]
type Metrics = dict[str, int | float | list[int] | list[float]]
type Extras  = dict[str, str | bool | int | float | bytes
               | list[str] | list[bool] | list[int] | list[float] | list[bytes]]


@dataclass(frozen=True)  # Can not replace top level dicts once created
class Update:
    arrays: dict[str, Arrays] = field(default_factory=dict)
    objects: dict[str, Objects] = field(default_factory=dict)
    metrics: dict[str, Metrics] = field(default_factory=dict)
    extras: dict[str, Extras] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (not self.arrays
                and not self.objects
                and not self.metrics
                and not self.extras)


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
