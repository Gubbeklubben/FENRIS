from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import fields
from typing import Annotated

import typer

from fenris.app.plugins import Group, plugins
from fenris.core.component import Component, Metadata

app = typer.Typer()


@app.command()
def show(
    groups: Annotated[
        list[str] | None,
        typer.Argument(
            ...,
            metavar=f"{'|'.join(plugins.groups.keys())} ...",
            help="Groups of components to show. If omitted, all are shown.",
            callback=validate_groups,
        ),
    ] = None,
    show_metadata: Annotated[
        bool,
        typer.Option(
            "--metadata",
            help="Show relevant metadata.",
        ),
    ] = False,
    keywords: Annotated[
        bool,
        typer.Option(
            "--keywords",
            help="Show valid constructor arguments and associated default values.",
        ),
    ] = False,
) -> None:
    """Show available components.

    Examples
    --------
      fenris show
      fenris show synthesizers
      fenris show synthesizers coordinators --metadata
    """
    selected = groups if groups else list(plugins.groups.keys())

    def show_evaluators() -> None:
        # Sort evaluator metadata by category
        evaluators_by_category = defaultdict(list)
        registry = plugins.evaluators.registry
        for metadata in registry.metadata():
            evaluator_cls = registry.load(metadata.name)
            spec = evaluator_cls.EVALUATOR_SPEC
            category = spec.category
            evaluators_by_category[category].append((metadata, spec))

        for category, entries in evaluators_by_category.items():
            # Category title
            title = f"{category.capitalize()} evaluators"
            typer.echo()
            typer.echo(f"{'':<2}{title}")
            typer.echo(f"{'':<2}{'\u2500' * len(title)}")

            for metadata, spec in entries:
                # Evaluator title
                typer.echo(f"{'':<4}{metadata.name}")

                if show_metadata:
                    _show_metadata(metadata, indent=6)

                    eval_mode = spec.eval_mode.name or ""
                    typer.echo(f"{'':<6}Evaluation mode: {eval_mode.lower()}")

                    typer.echo(f"{'':<6}Metrics:")
                    for metric in spec.metrics:
                        typer.echo(f"{'':<8}{metric.key}", nl=False)
                        typer.echo(
                            f".<{metric.suffix_type}_column>"
                            if metric.suffix_type
                            else ""
                        )
                    typer.echo()

    def _show(group: Group[Component]) -> None:
        # Component type heading
        typer.echo()
        typer.echo(group.name)
        typer.echo("\u2500" * len(group.name))

        if group is plugins.evaluators:
            show_evaluators()
            return

        registry = group.registry
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

    for name in selected:
        _show(plugins.groups[name])

    typer.echo()


def _show_metadata(metadata: Metadata, indent: int) -> None:
    for f in fields(metadata):
        if f.name in ("value", "group"):
            continue
        typer.echo(f"{'':<{indent}}{f.name}: {getattr(metadata, f.name)}")
    typer.echo()


def validate_groups(groups: list[str] | None) -> list[str] | None:
    if groups is None:
        return None

    for group in groups:
        if group not in plugins.groups:
            raise typer.BadParameter(f"'{group}' is not a valid group.")
    return groups
