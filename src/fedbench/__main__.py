import itertools
from typing import Annotated

import typer
from flwr.simulation import run_simulation

import fedbench.algorithms as algorithms
# noinspection PyProtectedMember
from fedbench._flwr import client_app
# noinspection PyProtectedMember
from fedbench._flwr import make_server_app

app = typer.Typer()


@app.command()
def new(name: str):
    pass


@app.command()
def show_algorithms(
        include_locator: Annotated[
            bool,
            typer.Option(
                "--include-locators",
                help="Show locators for the factories used to create "
                     "algorithm instances.")] = False) -> None:

    for name, locator in itertools.chain(algorithms.builtins(),
                                         algorithms.plugins()):
        print(name, end="")
        print(f": {locator}" if include_locator else "")


@app.command()
def run(
        algorithm_name: str,
        num_clients: int = typer.Option(default=3)) -> None:

    run_simulation(
        make_server_app(algorithm_name, num_clients),
        client_app,
        num_clients
    )


if __name__ == "__main__":
    app()
