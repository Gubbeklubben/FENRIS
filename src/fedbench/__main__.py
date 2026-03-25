from enum import StrEnum
from typing import Annotated, Literal

import typer

import fedbench.runtime.runner as runner
from fedbench.config.builder import build_config
from fedbench.config.parsing import split_outside_brackets
from fedbench.runtime.pipeline import pipeline
from fedbench.runtime.registry import FactoryRegistry
from fedbench.runtime.registry_builder import (
    build_algorithm_registry,
    build_coordinator_registry,
    build_evaluator_registry,
    build_partitioner_registry,
)

algorithms = build_algorithm_registry()
coordinators = build_coordinator_registry()
partitioners = build_partitioner_registry()
evaluators = build_evaluator_registry()

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


class Component(StrEnum):
    ALGORITHMS = "algorithms"
    COORDINATORS = "coordinators"
    PARTITIONERS = "partitioners"
    EVALUATORS = "evaluators"


@app.command()
def show(
    components: Annotated[
        list[Component] | None,
        typer.Argument(
            help="Components to show. If omitted, all are shown.",
        ),
    ] = None,
    include_locators: Annotated[
        bool,
        typer.Option(
            "--include-locators",
            help="Include factory locators (import paths) in the output.",
        ),
    ] = False,
) -> None:
    """
    Show available algorithms, partitioners, and/or evaluators.

    Examples:\n
      fedbench show\n
      fedbench show algorithms\n
      fedbench show algorithms partitioners --include-locators
    """

    selected = components if components else list(Component)

    def maybe_show[T](component: Component, registry: FactoryRegistry[T]) -> None:
        if component not in selected:
            return

        entries = list(registry.metadata())
        width = max(len(e.name) for e in entries)
        print(f"\n --- {component.value.upper()} ---")
        for metadata in entries:
            print(f"  {metadata.name:<{width}}", end="")
            print(f"  {metadata.locator}" if include_locators else "")

    maybe_show(Component.ALGORITHMS, algorithms)
    maybe_show(Component.COORDINATORS, coordinators)
    maybe_show(Component.PARTITIONERS, partitioners)
    maybe_show(Component.EVALUATORS, evaluators)

    print()


@app.command()
def run(
    algorithm: Annotated[str, typer.Argument(help="Algorithm/Generator key.")],
    coordinator: Annotated[str, typer.Argument(help="Coordinator key.")],
    partitioner: Annotated[str, typer.Argument(help="Partitioner key.")],
    dataset: Annotated[str, typer.Argument(help="Path to the dataset CSV.")],
    algorithm_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the algorithm (key=value)."
        ),
    ] = None,
    coordinator_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the coordinator (key=value)"
        ),
    ] = None,
    partitioner_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the partitioner (key=value)."
        ),
    ] = None,
    target_col: Annotated[
        str | None, typer.Option(help="Target column for utility/fairness metrics.")
    ] = None,
    sensitive_cols: Annotated[
        str | None,
        typer.Option(
            callback=parse_args,
            help="Comma-separated sensitive columns for fairness/AIA.",
        ),
    ] = None,
    run_categories: Annotated[
        str | None,
        typer.Option(
            callback=parse_args, help="Override specific metric categories to run."
        ),
    ] = None,
    early_stop: Annotated[
        bool | None, typer.Option(help="Enable threshold-based convergence.")
    ] = None,
    stop_metric: Annotated[
        str | None,
        typer.Option(help="Metric key to monitor (e.g., fidelity.corr_fro_diff)."),
    ] = None,
    stop_mode: Annotated[
        Literal["min", "max"] | None, typer.Option(help="min or max.")
    ] = None,
    stop_epsilon: Annotated[
        float | None, typer.Option(help="Minimum improvement required.")
    ] = None,
    stop_patience: Annotated[
        int | None,
        typer.Option(help="Evaluations without improvement before stopping."),
    ] = None,
    stop_min_rounds: Annotated[
        int | None, typer.Option(help="Minimum rounds before stopping.")
    ] = None,
    stop_eval_every: Annotated[
        int | None, typer.Option(help="Evaluate stopping metric every N rounds.")
    ] = None,
    stop_synthetic_rows: Annotated[
        int | None, typer.Option(help="Rows sampled for early stopping evaluation.")
    ] = None,
    num_clients: Annotated[
        int | None, typer.Option(help="Number of simulated clients.")
    ] = None,
    num_rounds: Annotated[
        int | None, typer.Option(help="Maximum number of federated rounds.")
    ] = None,
    test_size: Annotated[
        float | None, typer.Option(help="Fraction of data to hold out for testing.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Master random seed.")] = None,
    outputdir: Annotated[
        str | None, typer.Option(help="Output directory for artifacts.")
    ] = None,
    num_synthetic_rows: Annotated[
        int | None, typer.Option(help="Number of synthetic rows to generate.")
    ] = None,
    disable_pickle: Annotated[
        bool | None, typer.Option(help="Disable pickle for dataset loading.")
    ] = None,
) -> None:

    cli_input = {
        key: value  # nofmt
        for key, value in locals().items()
        if value is not None
    }
    config = build_config(cli_input, algorithms, partitioners)
    runner.run(config, pipeline())


if __name__ == "__main__":
    app()
