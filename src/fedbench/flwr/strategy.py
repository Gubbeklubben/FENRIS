from typing import cast

from flwr.server import Grid

from fedbench.core.algorithm import Aggregator
from fedbench.core.eventbus import EventBus
from fedbench.core.events import (
    AlgorithmInitStarted,
    AlgorithmInitCompleted,
    ServerRequest,
    ClientReply,
    RoundStarted,
    RoundCompleted,
)
from fedbench.core.update import Update, Metrics
from fedbench.flwr.serde import FlwrSerializer, FlwrDeserializer


class FedbenchStrategy:
    def __init__(
            self,
            eventbus: EventBus,
            aggregator: Aggregator,
            to_flwr: FlwrSerializer,
            from_flwr: FlwrDeserializer) -> None:

        self._eventbus = eventbus
        self._aggregator = aggregator
        self._to_flwr = to_flwr
        self._from_flwr = from_flwr
        self._prev_aggr_update: Update | None = None
        self._per_client_metrics: dict[int, Metrics] = {}

    def init(self, grid: Grid) -> Update:
        requests = []
        for cid, update in self._aggregator.configure_init(grid.get_node_ids()):
            # noinspection PyUnnecessaryCast
            request = self._to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="init",
                dst_node_id=cid
            )
            self._eventbus.emit(ServerRequest(cid, msg_type="init"))
            requests.append(request)

        arrays_map = self._aggregator.arrays_to_ml_framework_map
        replies = []
        for reply in grid.send_and_receive(requests):
            self._eventbus.emit(
                ClientReply(reply.metadata.src_node_id, msg_type="init")
            )
            replies.append(self._from_flwr(reply, arrays_map))

        return self._aggregator.aggregate_init(replies)

    def train(self, grid: Grid) -> Update:
        requests = []
        for cid in grid.get_node_ids():
            # noinspection PyUnnecessaryCast
            request = self._to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="train",
                dst_node_id=cid
            )
            self._eventbus.emit(ServerRequest(cid, msg_type="train"))
            requests.append(request)

        arrays_map = self._aggregator.arrays_to_ml_framework_map
        replies = []
        for reply in grid.send_and_receive(requests):
            self._eventbus.emit(
                ClientReply(reply.metadata.src_node_id, msg_type="train")
            )
            replies.append(self._from_flwr(reply, arrays_map))

        return self._aggregator.aggregate_train(replies)

    def evaluate(self, grid: Grid) -> None:
        requests = []
        for cid in grid.get_node_ids():
            # noinspection PyUnnecessaryCast
            request = self._to_flwr(
                cast(Update, self._prev_aggr_update),
                message_type="evaluate",
                dst_node_id=cid
            )
            self._eventbus.emit(ServerRequest(cid, msg_type="evaluate"))
            requests.append(request)

        for reply in grid.send_and_receive(requests):
            client_id = reply.metadata.src_node_id
            self._eventbus.emit(ClientReply(client_id, msg_type="evaluate"))
            metrics = reply.content.metric_records["metrics"]
            self._per_client_metrics[client_id] = dict(metrics)

    def run(
            self,
            grid: Grid,
            num_rounds: int) -> tuple[Update, dict[str, float]]:

        self._eventbus.emit(AlgorithmInitStarted())
        init_update = self.init(grid)

        if init_update is None or init_update.is_empty():
            raise RuntimeError("No initial algorithm state present.")

        self._prev_aggr_update = init_update
        self._eventbus.emit(AlgorithmInitCompleted())

        for curr_round in range(1, num_rounds + 1):
            self._eventbus.emit(RoundStarted(curr_round, num_rounds))

            update = self.train(grid)
            if update is not None:
                self._prev_aggr_update = update

            self.evaluate(grid)
            self._eventbus.emit(RoundCompleted(curr_round, num_rounds))

        return self._prev_aggr_update, {}
