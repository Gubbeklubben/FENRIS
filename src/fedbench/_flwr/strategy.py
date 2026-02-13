import time
from collections.abc import Iterable
from logging import INFO
from typing import cast, Callable

from flwr.common import MetricRecord, ArrayRecord, ConfigRecord, Message
from flwr.server import Grid

from fedbench._flwr.serde import from_flwr, to_flwr
from fedbench.algorithms import RegisteredAlgorithm
from fedbench.common import log, Update


class FedbenchStrategy:
    def __init__(self, algorithm: RegisteredAlgorithm) -> None:
        self._algorithm = algorithm
        self._na_protocol = algorithm.cls.requires_non_array_protocol()
        self._synth_aggregator = algorithm.cls.aggregator_factory()
        self._prev_aggr_update: Update | None = None

    def init(self, grid: Grid, num_clients: int) -> Update:
        client_ids = list(grid.get_node_ids())

        # Wait for clients to connect.
        # https://github.com/adap/flower/blob/main/examples/federated-kaplan-meier-fitter/examplefkm/server_app.py
        while len(client_ids) < num_clients:
            time.sleep(1)
            client_ids = list(grid.get_node_ids())

        # noinspection PyUnnecessaryCast
        requests = (
            to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="query.init",
                dst_node_id=cid,
                non_array_protocol=self._na_protocol
            )
            for cid, update in
            self._synth_aggregator.configure_init(client_ids)
        )
        replies = grid.send_and_receive(requests)

        return self._synth_aggregator.aggregate_init(
            from_flwr(reply) for reply in replies
        )

    def train(self, grid: Grid) -> Update:
        # noinspection PyUnnecessaryCast
        requests = (
            to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="train",
                dst_node_id=cid,
                non_array_protocol=self._na_protocol
            ) for cid in grid.get_node_ids()
        )
        replies = grid.send_and_receive(self._inject_config(requests))

        return self._synth_aggregator.aggregate_train(
            from_flwr(reply) for reply in replies
        )

    def evaluate(self, grid: Grid):  # type: ignore[no-untyped-def]
        pass

    def start(
        self,
        grid: Grid,
        num_clients: int,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> None:

        log(f"Starting {self.__class__.__name__}", (), level=INFO)

        init_update = self.init(grid, num_clients)

        if init_update is None or init_update.is_empty():
            raise RuntimeError("No initial synthesizer state present.")

        self._prev_aggr_update = init_update

        for curr_round in range(1, num_rounds + 1):
            log(
                self.__class__.__name__,
                (f"[ROUND {curr_round}/{num_rounds}]",),
                level=INFO
            )
            update = self.train(grid)
            if update is not None:
                self._prev_aggr_update = update

           # Evaluation...

    def _inject_config(self, messages: Iterable[Message]) -> Iterable[Message]:
        for message in messages:
            message.content["fedbench.config"] = ConfigRecord({
                "algorithm-name": self._algorithm.name,
            })
            yield message





