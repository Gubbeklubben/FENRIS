from typing import Iterable, Callable

from flwr.common import Message, MetricRecord, ArrayRecord, ConfigRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy, Result

from fedbench._flwr.serde import to_flwr_message, from_flwr_message
from fedbench.synthesizers import ServerComponent


class FedbenchStrategy(Strategy):
    def __init__(self, server_component: ServerComponent):
        self._server_component = server_component

    def configure_init(self, client_ids: Iterable[int]) -> Iterable[Message]:
        return (
            to_flwr_message(data_out, message_type="init") for data_out in
            self._server_component.configure_init(client_ids)
        )

    def aggregate_init(self, replies: Iterable[Message]) -> ArrayRecord:
        data_ins = (from_flwr_message(reply, is_client=False) for reply in replies)
        arrays, objects = self._server_component.aggregate_init(data_ins)
        # Do something with the eventual objects...
        return ArrayRecord(arrays)  # empty if arrays None


    def configure_train(
            self,
            server_round: int,
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
        return ()

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message]) -> MetricRecord | None:
        return None

    def summary(self) -> None:
        pass

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result: