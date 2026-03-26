from enum import StrEnum
from typing import Annotated, Literal
import json
import shutil
from pathlib import Path

import typer

import fedbench.runtime.runner as runner
from fedbench.config.builder import build_config
from fedbench.config.parsing import split_outside_brackets
from fedbench.runtime.pipeline import pipeline
from fedbench.runtime.registry import FactoryRegistry
from fedbench.runtime.registry_builder import (
    build_coordinator_registry,
    build_evaluator_registry,
    build_partitioner_registry,
    build_synthesizer_registry,
)

synthesizers = build_synthesizer_registry()
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
    SYNTHESIZERS = "synthesizers"
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
    Show available components.

    Examples:\n
      fedbench show\n
      fedbench show synthesizers\n
      fedbench show synthesizers partitioners --include-locators
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

    maybe_show(Component.SYNTHESIZERS, synthesizers)
    maybe_show(Component.COORDINATORS, coordinators)
    maybe_show(Component.PARTITIONERS, partitioners)
    maybe_show(Component.EVALUATORS, evaluators)

    print()

@app.command()
def configure() -> None:
    """
    Interactively configure a fedbench run and save it to a JSON file.
    """
    config = {}

    # Show available components
    typer.echo("\nAvailable synthesizers: " + ", ".join(m.name for m in synthesizers.metadata()))
    typer.echo("Available coordinators: " + ", ".join(m.name for m in coordinators.metadata()))
    typer.echo("Available partitioners: " + ", ".join(m.name for m in partitioners.metadata()))

    # Required positional arguments
    config["synthesizer"] = typer.prompt("\nSynthesizer")
    config["coordinator"] = typer.prompt("Coordinator")
    config["partitioner"] = typer.prompt("Partitioner")
    config["dataset"] = typer.prompt("Dataset path")

    # Optional kwargs
    synthesizer_kwargs_str = typer.prompt("Synthesizer kwargs (key=value,...)", default="")
    if synthesizer_kwargs_str:
        config["synthesizer_kwargs"] = parse_kwargs(synthesizer_kwargs_str)

    coordinator_kwargs_str = typer.prompt("Coordinator kwargs (key=value,...)", default="")
    if coordinator_kwargs_str:
        config["coordinator_kwargs"] = parse_kwargs(coordinator_kwargs_str)

    partitioner_kwargs_str = typer.prompt("Partitioner kwargs (key=value,...)", default="")
    if partitioner_kwargs_str:
        config["partitioner_kwargs"] = parse_kwargs(partitioner_kwargs_str)

    # Data options
    target_col = typer.prompt("Target column (leave blank to skip)", default="")
    if target_col:
        config["target_col"] = target_col

    sensitive_cols_str = typer.prompt("Sensitive columns (comma-separated, leave blank to skip)", default="")
    if sensitive_cols_str:
        config["sensitive_cols"] = sensitive_cols_str

    # Metrics options
    typer.echo("")
    run_categories_str = typer.prompt(
        "Run categories (comma-separated)",
        default="fidelity,utility,privacy,fairness,scalability",
    )
    if run_categories_str != "fidelity,utility,privacy,fairness,scalability":
        config["run_categories"] = run_categories_str

    if typer.confirm("Enable early stopping?", default=False):
        config["early_stop"] = True
        config["stop_metric"] = typer.prompt("Stop metric")
        config["stop_mode"] = typer.prompt("Stop mode (min/max)", default="min")
        config["stop_epsilon"] = float(typer.prompt("Stop epsilon", default=0.01))
        config["stop_patience"] = int(typer.prompt("Stop patience", default=5))
        config["stop_min_rounds"] = int(typer.prompt("Stop min rounds", default=1))
        config["stop_eval_every"] = int(typer.prompt("Stop eval every", default=1))
        stop_synthetic_rows = typer.prompt("Stop synthetic rows (leave blank to skip)", default="")
        if stop_synthetic_rows:
            config["stop_synthetic_rows"] = int(stop_synthetic_rows)

    # Run configuration
    typer.echo("")
    config["num_clients"] = int(typer.prompt("Number of clients", default=3))
    config["num_rounds"] = int(typer.prompt("Number of rounds", default=3))
    config["test_size"] = float(typer.prompt("Test size", default=0.2))
    config["seed"] = int(typer.prompt("Seed", default=42))

    outputdir = typer.prompt("Output directory (leave blank for default)", default="")
    if outputdir:
        config["outputdir"] = outputdir

    num_synthetic_rows = typer.prompt("Num synthetic rows (leave blank to skip)", default="")
    if num_synthetic_rows:
        config["num_synthetic_rows"] = int(num_synthetic_rows)

    if typer.confirm("Disable pickle?", default=False):
        config["disable_pickle"] = True

    # Validate
    available_synthesizers = {m.name for m in synthesizers.metadata()}
    available_coordinators = {m.name for m in coordinators.metadata()}
    available_partitioners = {m.name for m in partitioners.metadata()}

    if config["synthesizer"] not in available_synthesizers:
        typer.echo(f"Error: unknown synthesizer '{config['synthesizer']}'.", err=True)
        raise typer.Exit(code=1)

    if config["coordinator"] not in available_coordinators:
        typer.echo(f"Error: unknown coordinator '{config['coordinator']}'.", err=True)
        raise typer.Exit(code=1)

    if config["partitioner"] not in available_partitioners:
        typer.echo(f"Error: unknown partitioner '{config['partitioner']}'.", err=True)
        raise typer.Exit(code=1)

    # Save
    user_configs_dir = Path("user_configs")
    user_configs_dir.mkdir(exist_ok=True)

    output_path = typer.prompt("\nSave config as (e.g. config.json)")
    output_path = user_configs_dir / output_path

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    typer.echo(f"\nConfig saved to {output_path}")


@app.command()
def run(
    synthesizer: Annotated[str, typer.Argument(help="Synthesizer name.")],
    coordinator: Annotated[str, typer.Argument(help="Coordinator name.")],
    partitioner: Annotated[str, typer.Argument(help="Partitioner name.")],
    dataset: Annotated[str, typer.Argument(help="Path to the dataset CSV.")],
    synthesizer_kwargs: Annotated[
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
    config = build_config(cli_input, synthesizers, partitioners)
    runner.run(config, pipeline())



@app.command()
def run_from_config(
    config_file: Annotated[
        str,
        typer.Argument(help="Path to a JSON config file."),
    ],
) -> None:
    """
    Run a fedbench pipeline from a JSON config file.
    """
    with open(config_file) as f:
        cli_input = json.load(f)

    config = build_config(cli_input, synthesizers, partitioners)

    output_dir = Path(config.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(config_file).resolve()
    dst = (output_dir / Path(config_file).name).resolve()
    if src != dst:
        shutil.copy(config_file, dst)

    runner.run(config, pipeline())

if __name__ == "__main__":
    app()
