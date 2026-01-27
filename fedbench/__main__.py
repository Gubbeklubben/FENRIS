import typer
from flwr.simulation.run_simulation import run_simulation

from fedbench.flower.client import app as client_app
from fedbench.flower.server import app as server_app


app = typer.Typer()


@app.command()
def run() -> None:
    run_simulation(server_app, client_app, 10)


if __name__ == "__main__":
    app()
