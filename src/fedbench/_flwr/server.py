from flwr.common import Message, Context, RecordDict
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench._flwr.strategy import FedbenchStrategy
from fedbench.algorithms import load_algorithm


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

        algorithm = load_algorithm(algorithm_name)
        strategy = FedbenchStrategy(algorithm)
        strategy.start(grid, num_clients)

    return app