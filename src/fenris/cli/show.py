import inspect
from collections import defaultdict
from dataclasses import fields
from typing import Annotated

import typer

from fenris.app.registry import Group, Metadata

app = typer.Typer()


@app.command()
def show(
    groups: Annotated[
        list[Group] | None,
        typer.Argument(
            help="Groups of components to show. If omitted, all are shown.",
        ),
    ] = None,
    show_metadata: Annotated[
        bool,
        typer.Option(
            "--metadata",
            help="Include factory metadata in the output.",
        ),
    ] = False,
    keywords: Annotated[
        bool,
        typer.Option(
            "--keywords",
            help="Include valid factory keyword arguments and their default values in "
            "the output.",
        ),
    ] = False,
) -> None:
    """
    Show available components.

    Examples:
      fenris show
      fenris show synthesizers
      fenris show synthesizers coordinators --metadata
    """

    selected = groups if groups else list(Group)

    def show_evaluators() -> None:
        # Sort evaluator metadata by category
        evaluators_by_category = defaultdict(list)
        for evaluator in Group.EVALUATORS.get_registry().metadata():
            evaluator_factory = Group.EVALUATORS.get_registry().load(evaluator.name)
            try:
                metadata = evaluator_factory().metadata
                category = metadata.category
            except NotImplementedError:
                metadata = None
                category = "Uncategorized"
            evaluators_by_category[category].append((evaluator, metadata))

        for category, entries in evaluators_by_category.items():
            # Category title
            title = f"{category.capitalize()} evaluators"
            typer.echo()
            typer.echo(f"{'':<2}{title}")
            typer.echo(f"{'':<2}{'\u2500' * len(title)}")

            for metadata, evaluator_metadata in entries:
                # Evaluator title
                typer.echo(f"{'':<4}{metadata.name}")

                if show_metadata:
                    _show_metadata(metadata, indent=6)

                    if not evaluator_metadata:
                        typer.echo(
                            f"{'':<6}Error: Metadata missing for this evaluator."
                        )
                        continue

                    eval_mode = evaluator_metadata.eval_mode.name or ""
                    typer.echo(f"{'':<6}Evaluation mode: {eval_mode.lower()}")

                    typer.echo(f"{'':<6}Metrics:")
                    for metric in evaluator_metadata.metrics:
                        typer.echo(f"{'':<8}{metric.key}", end="")
                        typer.echo(
                            f".<{metric.suffix_type}_column>"
                            if metric.suffix_type
                            else ""
                        )
                    typer.echo()

    def maybe_show(group: Group) -> None:
        if group not in selected:
            return

        # Component type heading
        typer.echo()
        typer.echo(group.name)
        typer.echo("\u2500" * len(group.name))

        if group == Group.EVALUATORS:
            show_evaluators()
            return

        registry = group.get_registry()
        for metadata in registry.metadata():
            # Component name
            typer.echo(f"{'':<2}{metadata.name}")

            if show_metadata:
                _show_metadata(metadata, indent=4)

            if keywords:
                component_factory = registry.load(metadata.name)
                if params := inspect.signature(component_factory).parameters.values():
                    typer.echo(f"{'':<4}Parameters:")
                    for param in params:
                        typer.echo(f"{'':<6}{param}")
                    typer.echo()

    maybe_show(Group.SYNTHESIZERS)
    maybe_show(Group.COORDINATORS)
    maybe_show(Group.PARTITIONERS)
    maybe_show(Group.EVALUATORS)

    typer.echo()


def _show_metadata(metadata: Metadata, indent: int) -> None:
    for f in fields(metadata):
        if f.name in ("value", "group"):
            continue
        typer.echo(f"{'':<{indent}}{f.name}: {getattr(metadata, f.name)}")
    typer.echo()
