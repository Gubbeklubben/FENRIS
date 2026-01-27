from typing import Iterable

from flwr.common import Message, MetricRecord, ArrayRecord, ConfigRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy as FlowerStrategy

from fedbench.strategy import FedBenchStrategy


class StrategyAdapter(FlowerStrategy):
    def __init__(self, strategy: FedBenchStrategy) -> None:
        self._strategy = strategy

    def configure_train(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:
        pass


    def aggregate_train(
            self,
            server_round:
            int, replies:
            Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        pass

    def configure_evaluate(
            self,
            server_round:
            int, arrays: ArrayRecord,
            config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        pass

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message]) -> MetricRecord | None:
        pass

    def summary(self) -> None:
        pass