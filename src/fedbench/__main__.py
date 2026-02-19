import pprint
from pathlib import Path
from typing import Annotated, Literal

import typer
from flwr.simulation import run_simulation

from fedbench.config.builder import build_config
# noinspection PyProtectedMember
from fedbench.flwr import client_app
# noinspection PyProtectedMember
from fedbench.flwr import make_server_app
from fedbench.algorithms import registry as alg_registry
from fedbench.config import Config
from fedbench.config.config import DataConfig

app = typer.Typer()


def parse_kwargs(value: str) -> dict[str, int]:
    if value is None:
        return {}

    result = {}

    for item in value.split(","):
        key, val = item.split("=")
        result[key] = int(val)
    return result


@app.command()
def new(name: str) -> None:
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
        algorithm: Annotated[str, typer.Argument()],
        partitioner: Annotated[str, typer.Argument()],
        dataset: Annotated[str, typer.Argument()],
        algorithm_kwargs: Annotated[str | None, typer.Option(callback=parse_kwargs)] = None,
        partitioner_kwargs: Annotated[str | None, typer.Option(callback=parse_kwargs)] = None,
        target_col: Annotated[str | None, typer.Option()] = None,
        sensitive_cols: Annotated[str | None, typer.Option()] = None,

        run_categories: Annotated[str | None, typer.Option()] = None,
        early_stop: Annotated[bool | None, typer.Option()] = None,
        stop_metric: Annotated[str | None, typer.Option()] = None,
        stop_mode: Annotated[Literal["min", "max"] | None, typer.Option()] = None,
        stop_epsilon: Annotated[float | None, typer.Option()] = None,
        stop_patience: Annotated[int | None, typer.Option()] = None,
        stop_min_rounds: Annotated[int | None, typer.Option()] = None,
        stop_eval_every: Annotated[int | None, typer.Option()] = None,
        stop_synthetic_rows: Annotated[int | None, typer.Option()] = None,

        num_rounds: Annotated[int | None, typer.Option()] = None,
        test_size: Annotated[float | None, typer.Option()] = None,
        seed: Annotated[int | None, typer.Option()] = None,
        outputdir: Annotated[str | None, typer.Option()] = None,
        num_synthetic_rows: Annotated[int | None, typer.Option()] = None,
        allow_pickle: Annotated[bool | None, typer.Option()] = None,
) -> None:

    config_dict = {
        key: value
        for key, value in locals().items()
        if value is not None
    }

    config = build_config(config_dict)

    run_simulation(
        make_server_app(config),
        client_app,
        num_supernodes=config.num_clients,
    )


if __name__ == "__main__":
    app()
