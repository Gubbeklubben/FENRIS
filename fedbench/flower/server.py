from flwr.common import Context
from flwr.server import Grid
from flwr.serverapp import ServerApp


_strategy = None


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Initialization step somehow...
    # Start Flower strategy, possibly instantiating a FlwrStrategyAdapter
    pass