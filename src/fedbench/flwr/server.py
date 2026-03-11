import json
import time
from collections.abc import Generator, Iterable
from typing import Any, cast

from flwr.common import ConfigRecord, Context, Message, RecordDict
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench.config import Config
from fedbench.core.algorithm import Coordinator
from fedbench.core.data import TableSchema
from fedbench.core.eventbus import EventBus
from fedbench.core.events import (
    ClientReply,
    ClientsConfigured,
    FedInitCompleted,
    FedInitStarted,
    ServerRequest,
    TrainingCompleted,
    TrainingStarted,
)
from fedbench.core.runcontext import RunContext
from fedbench.core.update import Metrics, Update
from fedbench.flwr.serde import (
    FlwrDeserializer,
    FlwrSerializer,
    from_flwr_pickle,
    to_flwr_disable_pickle,
    to_flwr_pickle,
)


class FedbenchServer:
    def __init__(
        self,
        coordinator: Coordinator,
        seed: int,
        schema: TableSchema,
        to_flwr: FlwrSerializer,
        from_flwr: FlwrDeserializer,
        eventbus: EventBus,
    ) -> None:

        self._coordinator = coordinator
        self._seed = seed
        self._schema = schema
        self._to_flwr = to_flwr
        self._from_flwr = from_flwr
        self._eventbus = eventbus
        self._per_client_metrics: dict[int, Metrics] = {}

    def fed_init(self, grid: Grid) -> None:
        generator = self._coordinator.fed_init(
            self._seed,
            self._schema,
            grid.get_node_ids(),
        )
        self._send_and_receive(grid, generator, msg_type="query.init")

    def train(self, grid: Grid) -> None:
        generator = self._coordinator.train(grid.get_node_ids())
        self._send_and_receive(grid, generator, msg_type="train")

    def evaluate(self, grid: Grid) -> None:
        msg_type = "evaluate"
        global_state = self._get_and_check_global_state()
        requests = []

        for dst_id in grid.get_node_ids():
            request = self._to_flwr(
                global_state,
                message_type=msg_type,
                dst_node_id=dst_id,
            )
            requests.append(request)
            self._eventbus.emit(ServerRequest(dst_id, msg_type=msg_type))

        for reply in grid.send_and_receive(requests):
            src_id = reply.metadata.src_node_id
            self._eventbus.emit(ClientReply(src_id, msg_type=msg_type))
            metrics = reply.content.config_records["metrics"]
            # noinspection PyUnnecessaryCast
            self._per_client_metrics[src_id] = {
                key: json.loads(cast(str, value))  # nofmt
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
        arrays_map = self._coordinator.arrays_to_ml_framework_map
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
                request = self._to_flwr(
                    update,
                    message_type=msg_type,
                    dst_node_id=dst_id,
                )
                requests.append(request)
                self._eventbus.emit(ServerRequest(dst_id, msg_type=internal_msg_type))

            replies = []
            for reply in grid.send_and_receive(requests):
                src_id = reply.metadata.src_node_id
                replies.append((src_id, self._from_flwr(reply, arrays_map)))
                self._eventbus.emit(ClientReply(src_id, msg_type=internal_msg_type))

    def _get_and_check_global_state(self) -> Update:
        global_state = self._coordinator.global_state
        if not isinstance(global_state, Update):
            raise TypeError(
                f"{self._coordinator}.global_state returned"
                f"{type(global_state)}, expected {Update}"
            )
        return global_state


def configure_clients(grid: Grid, config: Config) -> Iterable[Message]:
    client_ids = list(grid.get_node_ids())

    # Wait for clients to connect.
    # https://github.com/adap/flower/blob/main/examples/federated-kaplan-meier-fitter/examplefkm/server_app.py
    while len(client_ids) < config.num_clients:
        time.sleep(1)
        client_ids = list(grid.get_node_ids())

    cfg_jsons = ConfigRecord({"jsons": config.jsons()})
    messages = (
        Message(
            content=RecordDict({"config": cfg_jsons}),
            message_type="query.configure",
            dst_node_id=cid,
        )
        for cid in client_ids
    )
    return grid.send_and_receive(messages)


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()
    config = ctx.config
    eventbus = ctx.eventbus
    algorithm = ctx.algorithm

    @app.main()
    def main(grid: Grid, _: Context) -> None:
        reply_count = 0
        for reply in configure_clients(grid, config):
            if reply.has_error():
                raise RuntimeError(
                    f"Failed to configure all clients: {reply.error.reason}"
                )
            reply_count += 1
        eventbus.emit(ClientsConfigured(reply_count))

        server = FedbenchServer(
            algorithm.create_coordinator(),
            config.seed,
            ctx.dataset.schema,
            to_flwr=to_flwr_disable_pickle if config.disable_pickle else to_flwr_pickle,
            from_flwr=from_flwr_pickle,
            eventbus=eventbus,
        )
        state, metrics = server.run(grid, config.num_rounds)
        ctx.aggregated_state = state
        ctx.per_client_metrics = metrics

    return app
