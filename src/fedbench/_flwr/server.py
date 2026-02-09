import time

from flwr.common import Context, Message, RecordDict, ConfigRecord, ArrayRecord
from flwr.server import Grid
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg

from fedbench.algorithms import load_factory as load_algorithm_factory
from fedbench.common import InitResponse, ConfigDict


def to_init_response(message: Message) -> InitResponse:
    record = message.content.array_records["init"]
    return InitResponse(
        client_id=message.metadata.src_node_id,
        statistics={k: arr.numpy() for k, arr in record.items()},
    )


# Capture commandline args in a closure as we can not easily
# inject into Context (?). Re-consider other more robust approaches later.
def make_server_app(
        algorithm_name: str,
        num_clients: int) -> ServerApp:

    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        # The plan(ish):
        # - Load and validate server policy if not already done.
        # - Use grid.send_and_receive to send an init_request,
        #   inject algorithm name in config to allow clients to import,
        #   and then receive init responses.
        # - Inject server_policy.init with init_responses to obtain, in any case
        #   initial model state.
        # - Resolve appropriate strategy, either an adapter or a flwr native.
        # - Call strategy.start, inject config from either cmdline or elsewhere.
        # - ...

        config: ConfigDict = {"algorithm-name": algorithm_name}
        init_messages: list[Message] = []

        # Wait for clients to connect.
        # Approach ripped from: https://github.com/adap/flower/blob/main/examples/federated-kaplan-meier-fitter/examplefkm/server_app.py
        client_ids = list(grid.get_node_ids())
        while len(client_ids) < num_clients:
            time.sleep(1)
            client_ids = list(grid.get_node_ids())

        # Initialization
        for client_id in client_ids:
            init_messages.append(
                Message(
                dst_node_id=client_id,
                message_type="query.init",  # routed to client_app.query("init")
                content=RecordDict({"config": ConfigRecord(config)}),
            ))
        replies = grid.send_and_receive(init_messages)

        factory = load_algorithm_factory(algorithm_name)
        algorithm = factory()
        init_model_state = algorithm.server_initialize(
            to_init_response(msg) for msg in replies
        )

        # Start federation loop
        strategy = FedAvg()
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(init_model_state),
        )
    return app