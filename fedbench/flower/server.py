from flwr.common import Context
from flwr.server import Grid
from flwr.serverapp import ServerApp


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    pass