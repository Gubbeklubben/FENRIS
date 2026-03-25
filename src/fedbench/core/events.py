from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

type ObserverEntry = tuple[Observer, Iterable[type[Event]]]


class Observer(Protocol):
    def __call__(self, event: Event) -> None:
        pass


@dataclass(frozen=True)
class Event:
    timestamp_ns: float = field(init=False, default_factory=time.perf_counter_ns)


@dataclass(frozen=True)
class RunStarted(Event):
    run_id: str


@dataclass(frozen=True)
class RunCompleted(Event):
    run_id: str


@dataclass(frozen=True)
class RunFailed(Event):
    run_id: str
    current_command: str
    error_type: str
    error_msg: str


@dataclass(frozen=True)
class CommandStarted(Event):
    name: str


@dataclass(frozen=True)
class CommandCompleted(Event):
    name: str


@dataclass(frozen=True)
class ClientsConfigured(Event):
    pass


@dataclass(frozen=True)
class RoundStarted(Event):
    current: int
    total: int


@dataclass(frozen=True)
class RoundCompleted(Event):
    current: int
    total: int


@dataclass(frozen=True)
class ServerRequest(Event):
    client_id: int
    msg_type: str
    byte_count: int


@dataclass(frozen=True)
class ClientReply(Event):
    client_id: int
    msg_type: str
    byte_count: int
