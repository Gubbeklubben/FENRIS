from logging import INFO
from typing import cast

from flwr.server import Grid

from fedbench._flwr.serde import from_flwr, to_flwr
from fedbench.algorithms import Aggregator
from fedbench.common import log, Update


class FedbenchStrategy:
    def __init__(
            self,
            synth_aggregator: Aggregator,
            na_protocol: str,
            num_rounds: int) -> None:

        self._synth_aggregator = synth_aggregator
        self._na_protocol = na_protocol
        self._num_rounds = num_rounds
        self._prev_aggr_update: Update | None = None

    def init(self, grid: Grid) -> Update:
        # noinspection PyUnnecessaryCast
        requests = (
            to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="init",
                dst_node_id=cid,
                non_array_protocol=self._na_protocol
            )
            for cid, update in
            self._synth_aggregator.configure_init(grid.get_node_ids())
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
        replies = grid.send_and_receive(requests)

        return self._synth_aggregator.aggregate_train(
            from_flwr(reply) for reply in replies
        )

    def evaluate(self, grid: Grid):  # type: ignore[no-untyped-def]
        pass

    def start(self, grid: Grid) -> None:
        init_update = self.init(grid)

        if init_update is None or init_update.is_empty():
            raise RuntimeError("No initial algorithm state present.")

        log(
            self.__class__.__name__,
            ("Initialization complete.",),
            level=INFO)

        self._prev_aggr_update = init_update

        for curr_round in range(1, self._num_rounds + 1):
            log(
                self.__class__.__name__,
                (f"Starting [ROUND {curr_round}/{self._num_rounds}]",),
                level=INFO)

            update = self.train(grid)
            if update is not None:
                self._prev_aggr_update = update

            log(
                "",
                ("Training complete.",
                 f"Algorithm state: {self._prev_aggr_update}",),
                level=INFO
            )
           # Evaluation...
