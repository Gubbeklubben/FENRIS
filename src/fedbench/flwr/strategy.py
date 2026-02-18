import pprint
from logging import INFO
from typing import cast

from flwr.server import Grid

from fedbench.flwr.serde import FlwrSerializer, FlwrDeserializer
from fedbench.algorithms import Aggregator
from fedbench.common import log, Update, Metrics


class FedbenchStrategy:
    def __init__(
            self,
            synth_aggregator: Aggregator,
            num_rounds: int,
            to_flwr: FlwrSerializer,
            from_flwr: FlwrDeserializer) -> None:

        self._synth_aggregator = synth_aggregator
        self._num_rounds = num_rounds
        self._to_flwr = to_flwr
        self._from_flwr = from_flwr
        self._prev_aggr_update: Update | None = None
        self._per_client_metrics: dict[int, Metrics] = {}

    def init(self, grid: Grid) -> Update:
        # noinspection PyUnnecessaryCast
        requests = (
            self._to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="init",
                dst_node_id=cid
            )
            for cid, update in
            self._synth_aggregator.configure_init(grid.get_node_ids())
        )
        replies = grid.send_and_receive(requests)
        arrays_map = self._synth_aggregator.arrays_to_ml_framework_map

        return self._synth_aggregator.aggregate_init(
            self._from_flwr(reply, arrays_map) for reply in replies
        )

    def train(self, grid: Grid) -> Update:
        # noinspection PyUnnecessaryCast
        requests = (
            self._to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="train",
                dst_node_id=cid
            ) for cid in grid.get_node_ids()
        )
        replies = grid.send_and_receive(requests)
        arrays_map = self._synth_aggregator.arrays_to_ml_framework_map

        return self._synth_aggregator.aggregate_train(
            self._from_flwr(reply, arrays_map) for reply in replies
        )

    def evaluate(self, grid: Grid) -> None:
        # noinspection PyUnnecessaryCast
        requests = (
            self._to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="evaluate",
                dst_node_id=cid
            ) for cid in grid.get_node_ids()
        )
        for reply in grid.send_and_receive(requests):
            client_id = reply.metadata.src_node_id
            metrics = reply.content.metric_records["metrics"]
            self._per_client_metrics[client_id] = dict(metrics)

    def start(self, grid: Grid) -> None:
        init_update = self.init(grid)

        if init_update is None or init_update.is_empty():
            raise RuntimeError("No initial algorithm state present.")

        self._prev_aggr_update = init_update
        log(self.__class__.__name__, ("Initialization complete.",), level=INFO)

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
            self.evaluate(grid)

        log(
            self.__class__.__name__,
            ("Federation loop complete.",
            pprint.pformat(self._per_client_metrics)),
            level=INFO)
