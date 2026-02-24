import time
from collections.abc import Iterable

from flwr.common import Context, Message, ConfigRecord, RecordDict
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench.config import Config
from fedbench.core.events import ClientsConfigured
from fedbench.core.runcontext import RunContext
from fedbench.flwr.serde import make_serde
from fedbench.flwr.strategy import FedbenchStrategy


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
            dst_node_id=cid
        ) for cid in client_ids
    )
    return grid.send_and_receive(messages)


def make_server_app(runcontext: RunContext) -> ServerApp:
    app = ServerApp()
    config = runcontext.config
    eventbus = runcontext.eventbus
    algorithm = runcontext.components.algorithm

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

        strategy = FedbenchStrategy(
            eventbus,
            algorithm.create_aggregator(),
            *make_serde(config.allow_pickle)
        )
        runcontext.final_aggregated_state = strategy.run(grid, config.num_rounds)

    return app