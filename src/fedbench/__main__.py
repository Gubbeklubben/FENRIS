import inspect
from typing import Annotated, Literal

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


if __name__ == "__main__":
    app()
