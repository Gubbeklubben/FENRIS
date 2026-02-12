from __future__ import annotations

from dataclasses import dataclass
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any

from flwr.common.logger import log as _flwr_log
from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type Arrays = list[NDArray] | dict[str, torch.Tensor]
type MetricDict = dict[str, int | float | list[int] | list[float]]
type ConfigDict = dict[str, str | bool | int | float | bytes
    | list[str] | list[bool] | list[int] | list[float] | list[bytes]]


class MessageContent:
    def __init__(self) -> None:
        self._arrays = {}
        self._objects = {}
        self._metrics = {}
        self._config = {}

    @property
    def arrays(self) -> dict[str, Arrays]:
        return self._arrays

    @property
    def objects(self) -> dict[str, Any]:
        return self._objects

    @property
    def metrics(self) -> dict[str, MetricDict]:
        return self._metrics

    @property
    def config(self) -> dict[str, ConfigDict]:
        return self._config

    def is_empty(self):
        return (not self._arrays
                and not self._objects
                and not self._metrics
                and not self._config)

    def add_arrays(self, key: str, arrays: Arrays) -> None:
        if key in self._arrays:
            raise ValueError(f"Arrays with key '{key}' already exist.")
        self._arrays[key] = arrays

    def add_objects(self, key: str, objects: dict[str, Any]) -> None:
        if key in self._objects:
            raise ValueError(f"Objects with key '{key}' already exist.")
        self._objects[key] = objects

    def add_metrics(self, key: str, metrics: MetricDict) -> None:
        if key in self._metrics:
            raise ValueError(f"Metrics with key '{key}' already exist.")
        self._metrics[key] = metrics

    def add_config(self, key: str, config: ConfigDict) -> None:
        if key in self._config:
            raise ValueError(f"Config with key '{key}' already exist.")
        self._config[key] = config


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
