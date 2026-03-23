import math

from fedbench.core.events import (
    ClientReply,
    Event,
    ServerRequest,
    TrainEvalLoopCompleted,
    TrainEvalLoopStarted,
)


class ScalabilityCollector:
    """Observes run events and accumulates scalability measurements."""

    def __init__(self) -> None:
        self.wall_clock_seconds: float = math.nan
        self.bytes_sent: int = 0
        self.bytes_received: int = 0
        self.rounds_to_converge: float = math.nan
        self._t_start: float | None = None

    def __call__(self, event: Event) -> None:
        if isinstance(event, TrainEvalLoopStarted):
            self._t_start = event.timestamp_ns
        elif isinstance(event, TrainEvalLoopCompleted):
            self.rounds_to_converge = event.rounds
            if self._t_start is not None:
                delta_ns = event.timestamp_ns - self._t_start
                self.wall_clock_seconds = delta_ns / 1e9
        elif isinstance(event, ServerRequest):
            self.bytes_sent += event.byte_count
        elif isinstance(event, ClientReply):
            self.bytes_received += event.byte_count

    def get_metrics(self) -> dict[str, float]:
        return {
            "scalability.wall_clock_seconds": self.wall_clock_seconds,
            "scalability.bytes_sent": self.bytes_sent,
            "scalability.bytes_received": self.bytes_received,
            "scalability.rounds_to_converge": self.rounds_to_converge,
        }
