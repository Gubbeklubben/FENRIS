from typing import Annotated, Literal

import typer

import fedbench.pipeline as pipeline
import fedbench.runner as runner
from fedbench.config.builder import build_config
from fedbench.core.eventbus import EventBus
from fedbench.core.events import Event
from fedbench.util.parsing import split_outside_brackets
from fedbench.registries import (
    build_algorithm_registry,
    build_partitioner_registry,
)

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
        sensitive_cols: Annotated[str | None, typer.Option()] = None,

        run_categories: Annotated[str | None, typer.Option(callback=split_outside_brackets)] = None,
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
    config = build_config(cli_input, algorithms, partitioners)
    eventbus = EventBus()
    def observer(event: Event) -> None:
        from fedbench.core.logging import log
        log(
            "observer",
            (f"observed {event}",)
        )
    eventbus.register(observer, (Event,))
    runner.run(config, eventbus, pipeline.default())


if __name__ == "__main__":
    app()
