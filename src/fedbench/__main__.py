from pathlib import Path
from typing import Annotated

import typer
from flwr.simulation import run_simulation

# noinspection PyProtectedMember
from fedbench.flwr import client_app
# noinspection PyProtectedMember
from fedbench.flwr import make_server_app
from fedbench.algorithms import registry as alg_registry
from fedbench.config import Config
from fedbench.config.config import DataConfig

app = typer.Typer()


@app.command()
def new(name: str):
    pass


@app.command()
def list_algorithms(
        include_locator: Annotated[
            bool,
            typer.Option(
                "--include-locators",
                help="Show locators for the factories used to create "
                     "algorithm instances.")] = False) -> None:

    for metadata in alg_registry:
        print(metadata.name, end="")
        print(f": {metadata.locator}" if include_locator else "")


@app.command()
def run(
        algorithm: str,
        dataset: str,
        num_clients: int = typer.Option()) -> None:

    config = _build_minimal_example_config(
        algorithm=algorithm,
        dataset=dataset,
        num_clients=num_clients
    )
    run_simulation(
        make_server_app(config),
        client_app,
        num_supernodes=num_clients,
    )


def _build_minimal_example_config(
        algorithm: str,
        dataset: str,
        num_clients: int) -> Config:

    dataset = Path(dataset).resolve()
    if not dataset.exists():
        raise ValueError(f"Dataset {dataset} does not exist.")

    if not dataset.is_file():
        raise ValueError(f"Dataset {dataset} is not a regular file.")

    return Config(
        algorithm=algorithm,
        num_clients=num_clients,
        num_rounds=3,
        test_size=0.2,
        seed=1337,
        outputdir=str(Path.cwd().joinpath("out")),
        data=DataConfig(
            dataset=str(dataset),
            partitioner="iid-partitioner",
            partitioner_kwargs={"num_partitions": num_clients},
        ),
        allow_pickle=True
    )


if __name__ == "__main__":
    app()
