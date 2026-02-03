import typer
from flwr.simulation import run_simulation

from fedbench.algorithm import Algorithm
from fedbench._plugins import algorithms, load_algorithm
# noinspection PyProtectedMember
from fedbench._flwr.server import make_server_app
# noinspection PyProtectedMember
from fedbench._flwr.client import app as client_app


app = typer.Typer()


@app.command()
def new(name: str):
    pass


@app.command()
def list_algorithms() -> None:
    print(*algorithms())


@app.command()
def run(
        algorithm_name: str,
        num_clients: int = typer.Option(default=3)) -> None:

    algorithm = load_algorithm(algorithm_name)
    assert isinstance(algorithm, Algorithm)

    run_simulation(
        make_server_app(algorithm_name, num_clients),
        client_app,
        num_clients
    )


if __name__ == "__main__":
    app()
