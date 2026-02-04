from collections.abc import Iterable

from flwr.common import Message, MetricRecord, ArrayRecord, ConfigRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy as FlwrStrategy

from fedbench.server_policy import ServerPolicy


# Adapt ServerPolicy implementations to Flower.
class ServerPolicyAdapter(FlwrStrategy):
    def __init__(self, server_policy: ServerPolicy) -> None:
        self._server_policy = server_policy

    def configure_train(
            self, server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:
        pass

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        pass

    def configure_evaluate(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:
        pass

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message]) -> MetricRecord | None:
        pass

    def summary(self) -> None:
        pass