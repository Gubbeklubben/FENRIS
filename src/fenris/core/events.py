from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

type ObserverEntry = tuple[Observer, Iterable[type[Event]]]


class Observer(Protocol):
    """Protocol for objects that receive framework events."""

    def __call__(self, event: Event) -> None:
        """Handle an event.

        Parameters
        ----------
        event : Event
            The event to handle.
        """


@dataclass(frozen=True)
class Event:
    """Base class for all framework events.

    Attributes
    ----------
    timestamp_ns : float
        High-resolution timestamp in nanoseconds when the event was created.
    """

    timestamp_ns: float = field(init=False, default_factory=time.perf_counter_ns)


@dataclass(frozen=True)
class RunStarted(Event):
    """Emitted when a benchmark run begins.

    Attributes
    ----------
    run_id : str
        Unique identifier for the run.
    """

    run_id: str


@dataclass(frozen=True)
class RunCompleted(Event):
    """Emitted when a benchmark run finishes successfully.

    Attributes
    ----------
    run_id : str
        Unique identifier for the run.
    """

    run_id: str


@dataclass(frozen=True)
class RunFailed(Event):
    """Emitted when a benchmark run terminates with an error.

    Attributes
    ----------
    run_id : str
        Unique identifier for the run.
    current_command : str
        Name of the command that was executing when the failure occurred.
    error_type : str
        Fully qualified exception type name.
    error_msg : str
        Exception message.
    """

    run_id: str
    current_command: str
    error_type: str
    error_msg: str


@dataclass(frozen=True)
class CommandStarted(Event):
    """Emitted when a pipeline command begins execution.

    Attributes
    ----------
    name : str
        Name of the command.
    """

    name: str


@dataclass(frozen=True)
class CommandCompleted(Event):
    """Emitted when a pipeline command finishes successfully.

    Attributes
    ----------
    name : str
        Name of the command.
    """

    name: str


@dataclass(frozen=True)
class ClientsConfigured(Event):
    """Emitted after all clients have been initialized for a run."""


@dataclass(frozen=True)
class TrainEvalLoopStarted(Event):
    """Emitted when the federated train/eval loop begins."""


@dataclass(frozen=True)
class TrainEvalLoopCompleted(Event):
    """Emitted when the federated train/eval loop finishes.

    Attributes
    ----------
    rounds : int
        Total number of training rounds that were executed.
    """

    rounds: int


@dataclass(frozen=True)
class RoundStarted(Event):
    """Emitted at the start of each federated training round.

    Attributes
    ----------
    current : int
        One-based index of the current round.
    total : int
        Total number of rounds scheduled.
    """

    current: int
    total: int


@dataclass(frozen=True)
class RoundCompleted(Event):
    """Emitted at the end of each federated training round.

    Attributes
    ----------
    current : int
        One-based index of the current round.
    total : int
        Total number of rounds scheduled.
    """

    current: int
    total: int


@dataclass(frozen=True)
class ServerRequest(Event):
    """Emitted when the server dispatches a request to a client.

    Attributes
    ----------
    client_id : int
        Identifier of the target client.
    msg_type : str
        Type label of the message.
    byte_count : int
        Serialised size of the request payload in bytes.
    """

    client_id: int
    msg_type: str
    byte_count: int


@dataclass(frozen=True)
class ClientReply(Event):
    """Emitted when a client response is received by the server.

    Attributes
    ----------
    client_id : int
        Identifier of the replying client.
    msg_type : str
        Type label of the message.
    byte_count : int
        Serialised size of the reply payload in bytes.
    """

    client_id: int
    msg_type: str
    byte_count: int
