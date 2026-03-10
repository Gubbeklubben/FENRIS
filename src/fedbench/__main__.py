from typing import Annotated, Literal, Optional

import typer

import fedbench.runner as runner
from fedbench.config.builder import build_config
from fedbench.pipeline import pipeline
from fedbench.registries import (
    build_algorithm_registry,
    build_partitioner_registry,
    build_evaluator_registries,
    Component,
)
from fedbench.util.parsing import split_outside_brackets

algorithms = build_algorithm_registry()
partitioners = build_partitioner_registry()
evaluators = build_evaluator_registries()

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
def show(
    components: Annotated[
        Optional[list[Component]],
        typer.Argument(
            help="Components to show. If omitted, all components are shown.",
        ),
    ] = None,
    include_locators: Annotated[
        bool,
        typer.Option(
            "--include-locators",
            help="Show locators for the factories used to create instances.",
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

    if Component.algorithms in selected:
        items = list(algorithms.metadata())
        width = max(len(m.name) for m in items)
        print("\n--- ALGORITHMS ---")
        for metadata in algorithms.metadata():
            print(f"  {metadata.name.ljust(width)}", end="")
            print(f"  {metadata.locator}" if include_locators else "")

    if Component.partitioners in selected:
        items = list(partitioners.metadata())
        width = max(len(m.name) for m in items)
        print("\n--- PARTITIONERS ---")
        for metadata in partitioners.metadata():
            print(f"  {metadata.name.ljust(width)}", end="")
            print(f"  {metadata.locator}" if include_locators else "")

    if Component.evaluators in selected:
        for category, registry in evaluators.items():
            print(f"\n--- EVALUATORS: {category.upper()} ---")
            registered_items = list(registry.metadata())
            if not registered_items:
                print("  (No evaluators registered)")
                continue
            width = max(len(m.name) for m in registered_items)
            for metadata in registered_items:
                print(f"  {metadata.name.ljust(width)}", end="")
                print(f"  {metadata.locator}" if include_locators else "")


@app.command()
def run(
    algorithm: Annotated[str, typer.Argument(help="Algorithm/Generator key.")],
    partitioner: Annotated[str, typer.Argument(help="Partitioner key.")],
    dataset: Annotated[str, typer.Argument(help="Path to the dataset CSV.")],
    algorithm_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the algorithm (key=value)."
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
