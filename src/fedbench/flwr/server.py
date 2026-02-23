import time
from collections.abc import Iterable
from logging import INFO

from flwr.common import Context, Message, ConfigRecord, RecordDict
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench.flwr.serde import make_serde
from fedbench.flwr.strategy import FedbenchStrategy
from fedbench.algorithms import Algorithm, registry as algorithm_reg
from fedbench.common import log
from fedbench.config import Config


def configure_clients(grid: Grid, config: Config) -> Iterable[Message]:
    num_clients = config.num_clients
    client_ids = list(grid.get_node_ids())

    # Wait for clients to connect.
    # https://github.com/adap/flower/blob/main/examples/federated-kaplan-meier-fitter/examplefkm/server_app.py
    while len(client_ids) < num_clients:
        time.sleep(1)
        client_ids = list(grid.get_node_ids())

    serialized_cfg = ConfigRecord({"jsons": config.jsons()})
    messages = (
        Message(
            content=RecordDict({"config": serialized_cfg}),
            message_type="query.configure",
            dst_node_id=cid
        ) for cid in client_ids
    )
    return grid.send_and_receive(messages)


# Capture config in a closure as we can not easily inject into Context (?).
def make_server_app(config: Config) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        for reply in configure_clients(grid, config):
            if reply.has_error():
                raise RuntimeError(
                    f"Failed to configure all clients: {reply.error.reason}"
                )
        log(__name__, (f"All clients configured.", ), level=INFO)

        algorithm: Algorithm = algorithm_reg.call(config.algorithm)
        aggregator = algorithm.create_aggregator()
        to_flwr, from_flwr = make_serde(config.allow_pickle)
        
        strategy = FedbenchStrategy(
            aggregator,
            config.num_rounds,
            to_flwr,
            from_flwr,
        )
        strategy.start(grid)

    return app