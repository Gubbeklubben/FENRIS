import sys
from typing import Annotated, Literal

import typer

import fedbench.runner as runner
from fedbench.config.builder import build_config
from fedbench.pipeline import pipeline
from fedbench.registries import (
    build_algorithm_registry,
    build_partitioner_registry,
)
from fedbench.util.parsing import split_outside_brackets

algorithms = build_algorithm_registry()
partitioners = build_partitioner_registry()

app = typer.Typer()


def parse_kwargs(value: str) -> dict[str, str]:
    if value is None:
        return {}

    result = {}

    for item in split_outside_brackets(value):
        key, val = item.split("=")
        result[key] = val
    return result


def parse_args(value: str) -> list[str]:
    if value is None:
        return []
    return value.split(",")


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

    for metadata in algorithms.metadata():
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
        sensitive_cols: Annotated[str | None, typer.Option(callback=parse_args)] = None,

        run_categories: Annotated[str | None, typer.Option(callback=parse_args)] = None,
        early_stop: Annotated[bool | None, typer.Option()] = None,
        stop_metric: Annotated[str | None, typer.Option()] = None,
        stop_mode: Annotated[Literal["min", "max"] | None, typer.Option()] = None,
        stop_epsilon: Annotated[float | None, typer.Option()] = None,
        stop_patience: Annotated[int | None, typer.Option()] = None,
        stop_min_rounds: Annotated[int | None, typer.Option()] = None,
        stop_eval_every: Annotated[int | None, typer.Option()] = None,
        stop_synthetic_rows: Annotated[int | None, typer.Option()] = None,

        num_clients: Annotated[int | None, typer.Option()] = None,
        num_rounds: Annotated[int | None, typer.Option()] = None,
        test_size: Annotated[float | None, typer.Option()] = None,
        seed: Annotated[int | None, typer.Option()] = None,
        outputdir: Annotated[str | None, typer.Option()] = None,
        num_synthetic_rows: Annotated[int | None, typer.Option()] = None,
        allow_pickle: Annotated[bool | None, typer.Option()] = None,
) -> None:

    cli_input = {
        key: value for key, value in locals().items() if value is not None
    }
    # noinspection PyBroadException
    try:
        config = build_config(cli_input, algorithms, partitioners)
    except Exception as exc:
        print(f"Failed to build valid config: {exc}", file=sys.stderr)
        sys.exit(1)
        
    runner.run(config, pipeline())


if __name__ == "__main__":
    app()
