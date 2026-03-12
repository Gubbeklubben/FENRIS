import json
import time
from collections.abc import Generator, Iterable
from typing import Any, cast

from flwr.common import ConfigRecord, Message, RecordDict
from flwr.server import Grid

from fedbench.config import Config
from fedbench.core.algorithm import Coordinator
from fedbench.core.data import TableSchema
from fedbench.core.encoder import FedbenchEncoder
from fedbench.core.events import (
    ClientReply,
    FedInitCompleted,
    FedInitStarted,
    ServerRequest,
    TrainingCompleted,
    TrainingStarted,
)
from fedbench.core.update import Metrics, Update
from fedbench.flwr.serde import (
    FlwrDeserializer,
    FlwrSerializer,
)
from fedbench.runtime.eventbus import EventBus


class Strategy:
    def __init__(
        self,
        seed: int,
        schema: TableSchema,
        to_flwr: FlwrSerializer,
        from_flwr: FlwrDeserializer,
        eventbus: EventBus,
        coordinator: Coordinator,
        arrays_to_ml_framework_map: dict[str, str] | None,
    ) -> None:

        self._seed = seed
        self._schema = schema
        self._to_flwr = to_flwr
        self._from_flwr = from_flwr
        self._eventbus = eventbus
        self._coordinator = coordinator
        self._arrays_map = arrays_to_ml_framework_map
        self._per_client_metrics: dict[int, Metrics] = {}

    def fed_init(self, grid: Grid) -> None:
        generator = self._coordinator.fed_init(
            self._seed,
            self._schema,
            grid.get_node_ids(),
        )
        self._send_and_receive(grid, generator, msg_type="query.fed_init")

    def train(self, grid: Grid) -> None:
        generator = self._coordinator.train(grid.get_node_ids())
        self._send_and_receive(grid, generator, msg_type="train")

    def evaluate(self, grid: Grid) -> None:
        msg_type = "evaluate"
        global_state = self._get_and_check_global_state()
        requests = []

        for dst_id in grid.get_node_ids():
            rdict = self._to_flwr(global_state)
            requests.append(
                Message(content=rdict, message_type=msg_type, dst_node_id=dst_id)
            )
            self._eventbus.emit(ServerRequest(dst_id, msg_type=msg_type))

        for reply in grid.send_and_receive(requests):
            src_id = reply.metadata.src_node_id
            self._eventbus.emit(ClientReply(src_id, msg_type=msg_type))
            metrics = reply.content.config_records["metrics"]
            # noinspection PyUnnecessaryCast
            self._per_client_metrics[src_id] = {
                key: json.loads(
                    cast(str, value),
                    object_hook=FedbenchEncoder.decode,
                )
                for key, value in metrics.items()
            }

    def run(
        self,
        grid: Grid,
        num_rounds: int,
    ) -> tuple[Update, dict[int, Any]]:

        self._eventbus.emit(FedInitStarted())
        self.fed_init(grid)
        self._eventbus.emit(FedInitCompleted())

        for curr_round in range(1, num_rounds + 1):
            self._eventbus.emit(TrainingStarted(curr_round, num_rounds))
            self.train(grid)
            self._eventbus.emit(TrainingCompleted(curr_round, num_rounds))
            self.evaluate(grid)

        return self._get_and_check_global_state(), self._per_client_metrics

    def _send_and_receive(
        self,
        grid: Grid,
        generator: Generator[
            Iterable[tuple[int, Update]],
            Iterable[tuple[int, Update]],
            None,
        ],
        msg_type: str,
    ) -> None:

        internal_msg_type = msg_type.split(".")[-1]
        replies: Iterable[tuple[int, Update]] | None = None

        while True:
            try:
                if replies is None:
                    batch = next(generator)
                else:
                    batch = generator.send(replies)
            except StopIteration:
                return

            requests = []
            for dst_id, update in batch:
                rdict = self._to_flwr(update)
                requests.append(
                    Message(content=rdict, message_type=msg_type, dst_node_id=dst_id)
                )
                self._eventbus.emit(ServerRequest(dst_id, msg_type=internal_msg_type))

            replies = []
            for reply in grid.send_and_receive(requests):
                src_id = reply.metadata.src_node_id
                replies.append(
                    (src_id, self._from_flwr(reply.content, self._arrays_map))
                )
                self._eventbus.emit(ClientReply(src_id, msg_type=internal_msg_type))

    def _get_and_check_global_state(self) -> Update:
        global_state = self._coordinator.global_state
        if not isinstance(global_state, Update):
            raise TypeError(
                f"{self._coordinator}.global_state returned"
                f"{type(global_state)}, expected {Update}"
            )
        return global_state


def send_config(grid: Grid, config: Config) -> Iterable[Message]:
    # Wait for clients to connect.
    # ref. flwr.serverapp.strategy.strategy_utils:sample_nodes
    client_ids = list(grid.get_node_ids())
    while len(client_ids) < config.num_clients:
        time.sleep(1)
        client_ids = list(grid.get_node_ids())

    messages = (
        Message(
            content=RecordDict({"config": ConfigRecord({"jsons": config.jsons()})}),
            message_type="query.config",
            dst_node_id=cid,
        )
        for cid in client_ids
    )
    return grid.send_and_receive(messages)


def send_artifacts(
    grid: Grid,
    to_flwr: FlwrSerializer,
    synthesizer_artifacts: Update | None,
) -> Iterable[Message]:

    artifacts = synthesizer_artifacts or Update()

    messages = (
        Message(content=to_flwr(artifacts), message_type="query.artifacts",
                dst_node_id=cid)
        for cid in grid.get_node_ids()
    )
    return grid.send_and_receive(messages)
