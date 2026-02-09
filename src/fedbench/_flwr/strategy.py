from collections.abc import Iterable

from flwr.common import Message, MetricRecord, ArrayRecord, ConfigRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy as FlwrStrategy

from fedbench.algorithms.algorithm import Algorithm
from fedbench.common import log_calls


# Adapt server side logic to Flower.
class Strategy(FlwrStrategy):
    def __init__(self, algorithm: Algorithm) -> None:
        self._algorithm = algorithm

    @log_calls
    def configure_train(
            self, server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:
        pass

    @log_calls
    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        pass

    @log_calls
    def configure_evaluate(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:
        return ()

    @log_calls
    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message]) -> MetricRecord | None:
        return None

    @log_calls
    def summary(self) -> None:
        return None