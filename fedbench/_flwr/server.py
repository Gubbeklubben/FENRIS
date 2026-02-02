from flwr.common import Context
from flwr.server import Grid
from flwr.serverapp import ServerApp


_server_policy = None


# Capture commandline args in a closure as we can not easily
# inject into Context. Re-consider other more robust approaches later.

def make_server_app(algorithm_name: str) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        # - Load and validate server policy if not already done.
        # - Use grid.send_and_receive to send an init_request,
        #   inject algorithm name in config to allow clients to import,
        #   and then receive init responses.
        # - Inject server_policy.init with init_responses to obtain, in any case
        #   initial model state.
        # - Resolve appropriate strategy, either an adapter or a flwr native.
        # - Call strategy.start, inject config from either cmdline or elsewhere.
        # - ...
        pass

    return app