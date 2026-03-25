import json
import time
from collections.abc import Generator, Iterable
from typing import Any, Self, cast

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid

from fedbench.config import Config, SeedConfig
from fedbench.core.algorithm import Coordinator
from fedbench.core.data import TableSchema
from fedbench.core.encoder import FedbenchEncoder
from fedbench.core.events import (
    ClientReply,
    RoundCompleted,
    RoundStarted,
    ServerRequest,
)
from fedbench.core.payload import Metrics, Payload
from fedbench.flwr.namespace import Namespace
from fedbench.flwr.serde import FlwrSerde, count_rdict_bytes
from fedbench.runtime.eventbus import EventBus


class Strategy:
    @classmethod
    def from_seed_config(
        cls,
        seed_config: SeedConfig,
        schema: TableSchema,
        serde: FlwrSerde,
        eventbus: EventBus,
        coordinator: Coordinator,
    ) -> Self:

        return cls(
            seed_config.init,
            schema,
            serde,
            eventbus,
            coordinator,
        )

    def __init__(
        self,
        init_seed: int,
        schema: TableSchema,
        serde: FlwrSerde,
        eventbus: EventBus,
        coordinator: Coordinator,
    ) -> None:

        self._init_seed = init_seed
        self._schema = schema
        self._serde = serde
        self._eventbus = eventbus
        self._coordinator = coordinator
        self._per_client_metrics: dict[int, Metrics] = {}

    def train(self, grid: Grid) -> None:
        generator = self._coordinator.train(grid.get_node_ids())
        self._send_and_receive(grid, generator, msg_type="train")

    def evaluate(self, grid: Grid) -> None:
        msg_type = "evaluate"
        global_state = self._get_and_check_global_state()
        requests = []

        for dst_id in grid.get_node_ids():
            rdict = self._serde.to_flwr(global_state)
            requests.append(
                Message(content=rdict, message_type=msg_type, dst_node_id=dst_id)
            )
            self._eventbus.emit(
                ServerRequest(
                    dst_id,
                    msg_type=msg_type,
                    byte_count=count_rdict_bytes(rdict),
                )
            )

        for reply in grid.send_and_receive(requests):
            src_id = reply.metadata.src_node_id
            self._eventbus.emit(
                ClientReply(
                    src_id,
                    msg_type=msg_type,
                    byte_count=count_rdict_bytes(reply.content),
                )
            )
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
    ) -> tuple[Payload, dict[int, Any]]:

        for curr_round in range(1, num_rounds + 1):
            self._eventbus.emit(RoundStarted(curr_round, num_rounds))
            self.train(grid)
            self.evaluate(grid)
            self._eventbus.emit(RoundCompleted(curr_round, num_rounds))

        return self._get_and_check_global_state(), self._per_client_metrics

    def _send_and_receive(
        self,
        grid: Grid,
        generator: Generator[
            Iterable[tuple[int, Payload]],
            Iterable[tuple[int, Payload]],
            None,
        ],
        msg_type: str,
    ) -> None:

        internal_msg_type = msg_type.split(".")[-1]
        replies: Iterable[tuple[int, Payload]] | None = None

        while True:
            try:
                if replies is None:
                    batch = next(generator)
                else:
                    batch = generator.send(replies)
            except StopIteration:
                return

            requests = []
            for dst_id, payload in batch:
                rdict = self._serde.to_flwr(payload)
                requests.append(
                    Message(content=rdict, message_type=msg_type, dst_node_id=dst_id)
                )
                self._eventbus.emit(
                    ServerRequest(
                        dst_id,
                        msg_type=internal_msg_type,
                        byte_count=count_rdict_bytes(rdict),
                    )
                )

            replies = []
            for reply in grid.send_and_receive(requests):
                src_id = reply.metadata.src_node_id
                replies.append((src_id, self._serde.from_flwr(reply.content)))

                self._eventbus.emit(
                    ClientReply(
                        src_id,
                        msg_type=internal_msg_type,
                        byte_count=count_rdict_bytes(reply.content),
                    )
                )

    def _get_and_check_global_state(self) -> Payload:
        global_state = self._coordinator.global_state
        if not isinstance(global_state, Payload):
            raise TypeError(
                f"{self._coordinator}.global_state returned"
                f"{type(global_state)}, expected {Payload}"
            )
        return global_state


def configure_clients(
    config: Config,
    artifacts: Payload | None,
    serde: FlwrSerde,
    grid: Grid,
) -> None:

    # Wait for clients to connect.
    # ref. flwr.serverapp.strategy.strategy_utils:sample_nodes
    client_ids = list(grid.get_node_ids())
    while len(client_ids) < config.num_clients:
        time.sleep(1)
        client_ids = list(grid.get_node_ids())

    content = RecordDict()

    framework_view = Namespace.FRAMEWORK.view(content)
    artifacts_view = Namespace.GLOBAL_INIT_ARTIFACTS.view(content)

    framework_view["config"] = ConfigRecord({"jsons": config.jsons()})

    if artifacts is not None:
        artifacts_view.update(serde.to_flwr(artifacts))

    requests = (
        Message(content=content, message_type="query.configure", dst_node_id=cid)
        for cid in client_ids
    )
    for reply in grid.send_and_receive(requests):
        if reply.has_error():
            raise RuntimeError(
                f"Failed to configure client {reply.metadata.src_node_id}: "
                f"{reply.error.reason}"
            )
