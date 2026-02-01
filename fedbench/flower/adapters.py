from collections.abc import Iterable

from flwr.common import Message, MetricRecord, ArrayRecord, ConfigRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy as FlwrStrategy

from fedbench.server.server_policy import ServerPolicy


def config_record_to_dict(config: ConfigRecord) -> dict[str, bool | int | float | bytes]:
    return dict(config)


def dict_to_config_record(config: dict[str, bool | int | float | bytes ]) -> ConfigRecord:
    return ConfigRecord(config)


def metric_record_to_dict(metrics: MetricRecord) -> dict[str, float]:
    return dict(metrics)


def dict_to_metric_record(metrics: dict[str, float]) -> MetricRecord:
    return MetricRecord(metrics)


# Adapt ServerPolicy implementations to Flower
# Convert Flower ArrayRecords to/from the registered ml_runtime (numpy/torch)
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


class FlwrStrategyDecorator(FlwrStrategy):
    def __init__(
            self,
            flwr_strategy_factory: FlwrStrategyFactory,
            configure_init = None,
            aggregate_init = None) -> None:

        self._flwr_strategy = flwr_strategy_factory()
        self._configure_init = configure_init
        self._aggregate_init = aggregate_init

    def configure_train(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:

        return self._flwr_strategy.configure_train(
            server_round, arrays, config, grid)

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:

        return self._flwr_strategy.aggregate_train(server_round, replies)

    def configure_evaluate(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid) -> Iterable[Message]:

        return self._flwr_strategy.configure_evaluate(
            server_round, arrays, config, grid)

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message]) -> MetricRecord | None:

        return self._flwr_strategy.aggregate_evaluate(server_round, replies)

    def summary(self) -> None:
        return self._flwr_strategy.summary()