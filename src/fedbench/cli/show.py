import inspect
from dataclasses import fields
from typing import Annotated

import typer

from fedbench.core.eval import Category, EvaluatorDescriptor
from fedbench.runtime.registry import Group, Metadata

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
      fedbench show
      fedbench show synthesizers
      fedbench show synthesizers coordinators --metadata
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

                if show_metadata:
                    _show_metadata(metadata, indent=6)

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

            if show_metadata:
                _show_metadata(metadata, indent=4)

            if keywords:
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


def _show_metadata(metadata: Metadata, indent: int) -> None:
    for f in fields(metadata):
        if f.name in ("value", "group"):
            continue
        print(f"{'':<{indent}}{f.name}: {getattr(metadata, f.name)}")
    print()
