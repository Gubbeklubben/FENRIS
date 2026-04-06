import inspect
from typing import Annotated, Literal
import json
import shutil
from pathlib import Path

import typer

import fedbench.runtime.runner as runner
from fedbench.config.builder import build_config
from fedbench.config.parsing import split_outside_brackets
from fedbench.core.eval import Category
from fedbench.core.eval.evaluator import EvaluatorDescriptor
from fedbench.runtime.pipeline import pipeline
from fedbench.runtime.registry import Group, Metadata

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
    groups: Annotated[
        list[Group] | None,
        typer.Argument(
            help="Groups of components to show. If omitted, all are shown.",
        ),
    ] = None,
    include_locators: Annotated[
        bool,
        typer.Option(
            "--include-locators",
            help="Include factory locators (import paths) in the output.",
        ),
    ] = False,
    include_details: Annotated[
        bool,
        typer.Option(
            "--include-details",
            help="Include keyword arguments or other detailed component information "
            "in the output.",
        ),
    ] = False,
) -> None:
    """
    Show available components.

    Examples:
      fedbench show
      fedbench show synthesizers
      fedbench show synthesizers coordinators --include-locators
      fedbench show partitioners evaluators --include-details
    """

    selected = groups if groups else list(Group)

    def show_evaluators() -> None:
        # Sort evaluator metadata by category
        evaluators_by_category: dict[
            Category, list[tuple[Metadata, EvaluatorDescriptor]]
        ] = {category: [] for category in Category}
        for evaluator in Group.EVALUATORS.get_registry().metadata():
            evaluator_factory = Group.EVALUATORS.get_registry().load(evaluator.name)
            metadata = evaluator_factory().metadata
            evaluators_by_category[metadata.category].append((evaluator, metadata))

        for category, entries in evaluators_by_category.items():
            # Category title
            title = f"{category.capitalize()} evaluators"
            print()
            print(f"{'':<2}{title}")
            print(f"{'':<2}{'\u2500' * len(title)}")

            for metadata, evaluator_metadata in entries:
                # Evaluator title
                print(f"{'':<4}{metadata.name}")

                if include_locators:
                    print(f"{'':<6}Locator: {metadata.locator}")

                if include_details:
                    eval_mode = evaluator_metadata.eval_mode.name or ""
                    print(f"{'':<6}Evaluation mode: {eval_mode.lower()}")

                    print(f"{'':<6}Metrics:")
                    for metric in evaluator_metadata.metrics:
                        print(f"{'':<8}{metric.key}", end="")
                        print(
                            f".<{metric.suffix_type}_column>"
                            if metric.suffix_type
                            else ""
                        )
                    print()

    def maybe_show(group: Group) -> None:
        if group not in selected:
            return

        # Component type heading
        print()
        print(group.name)
        print("\u2500" * len(group.name))

        if group == Group.EVALUATORS:
            show_evaluators()
            return

        registry = group.get_registry()
        for metadata in registry.metadata():
            # Component name
            print(f"{'':<2}{metadata.name}")

            if include_locators:
                print(f"{'':<4}Locator: {metadata.locator}")

            if include_details:
                component_factory = registry.load(metadata.name)
                if params := inspect.signature(component_factory).parameters.values():
                    print(f"{'':<4}Parameters:")
                    for param in params:
                        print(f"{'':<6}{param}")
                    print()

    maybe_show(Group.SYNTHESIZERS)
    maybe_show(Group.COORDINATORS)
    maybe_show(Group.PARTITIONERS)
    maybe_show(Group.EVALUATORS)

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
            callback=parse_kwargs, help="Kwargs for the synthesizer (key=value)."
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
            help="Comma-separated list of sensitive columns for fairness/AIA metrics.",
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
        typer.Option(help="Metric key to monitor (e.g., `fidelity.corr_fro_diff`)."),
    ] = None,
    stop_mode: Annotated[
        Literal["min", "max"] | None,
        typer.Option(
            help="Whether stop metric is expected to converge "
            "towards a minimum or a maximum."
        ),
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
        float | None,
        typer.Option(
            help="Fraction of data to hold out for testing. "
            "This is used for both local and global holdout fractions."
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Master random seed.")] = None,
    outputdir: Annotated[
        str | None, typer.Option(help="Output directory for artifacts.")
    ] = None,
    num_synthetic_rows: Annotated[
        int | None, typer.Option(help="Number of synthetic rows to generate.")
    ] = None,
    disable_pickle: Annotated[
        bool | None, typer.Option(help="Whether to disable pickle for dataset loading.")
    ] = None,
) -> None:

    cli_input = {
        key: value  # nofmt
        for key, value in locals().items()
        if value is not None
    }
    config = build_config(cli_input)
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
