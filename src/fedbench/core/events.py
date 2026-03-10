import time
from dataclasses import dataclass, field


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
    num_clients: int


@dataclass(frozen=True)
class FedInitStarted(Event):
    pass


@dataclass(frozen=True)
class FedInitCompleted(Event):
    pass


@dataclass(frozen=True)
class TrainingStarted(Event):
    current: int
    total: int


@dataclass(frozen=True)
class TrainingCompleted(Event):
    current: int
    total: int


@dataclass(frozen=True)
class ServerRequest(Event):
    client_id: int
    msg_type: str


@dataclass(frozen=True)
class ClientReply(Event):
    client_id: int
    msg_type: str
