from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, cast

import tomlkit
import typer
from tomlkit import TOMLDocument
from tomlkit.items import Table

from fenris.app.registry import Group
from fenris.cli.plugin._util import validate_identifier
from fenris.core.component import Component

app = typer.Typer()


@app.command()
def extend(
    plugin: Annotated[
        Path,
        typer.Argument(
            help="The plugin to extend.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            callback=_validate_plugin,
        ),
    ],
    group: Annotated[
        Group,
        typer.Argument(
            help="The group to extend.",
            is_eager=True,  # _validate_base needs group to have been parsed
        ),
    ],
    names: Annotated[
        list[str],
        typer.Argument(
            help="Names for which to generate scaffold implementations.",
            callback=_validate_and_normalize_names,
        ),
    ],
    package: Annotated[
        str | None,
        typer.Option(
            help="The subpackage in which to put generated modules. Accepts dot "
            "separated python identifiers.",
            callback=_validate_package,
        ),
    ] = None,
    base: Annotated[
        str | None,
        typer.Option(
            help="The ABC to implement, if there is more than one option.",
            callback=_validate_base,
        ),
    ] = None,
    no_group: Annotated[
        bool,
        typer.Option(
            "--no-group",
            help="Pass this flag if the default behaviour of grouping components by "
            "entry point is not desirable.",
        ),
    ] = False,
) -> None:
    """Add new components to an existing plugin project."""

    packages = [plugin.stem.lower()]
    if not no_group:
        packages.append(group.value)
    if package is not None:
        packages.extend(package.split("."))

    py_proj = plugin.joinpath("pyproject.toml")
    with py_proj.open("r") as f:
        toml = tomlkit.load(f)

    root_pkg = plugin.joinpath("src").joinpath(packages[0])
    cls = _get_class(group, base)
    try:
        entry_point = _ensure_entry_point(group, toml)
    except TypeError as exc:
        typer.echo(f"Error in {py_proj}: {str(exc)}", file=sys.stderr)
        raise typer.Abort()

    # Triggers some relatively heavy libcst imports. Importing here
    # makes the cli a little more responsive.
    from fenris.app.scaffold import create_component_scaffold

    for name in names:
        path = _descend_and_create_as_needed(root_pkg, packages[1:])
        path = path.joinpath(name.lower()).with_suffix(".py")
        if path.exists():
            typer.echo(f"File {path} already exists.")
            continue

        with path.open("w") as f:
            code = create_component_scaffold(cls, name, _to_cap_words(name))
            f.write(code)

        qualifier = f"{'.'.join((*packages, name))}:{_to_cap_words(name)}"
        entry_point[name] = qualifier

    with py_proj.open("w") as f:
        tomlkit.dump(toml, f)


def _ensure_entry_point(group: Group, toml: TOMLDocument) -> Table:
    project = toml["project"]
    if not isinstance(project, Table):
        raise TypeError(f"{project} is not a toml Table.")

    try:
        entry_points = project["entry-points"]
    except KeyError:
        entry_points = tomlkit.table()
        project["entry-points"] = entry_points

    if not isinstance(entry_points, Table):
        raise TypeError(f"{entry_points} is not a toml Table.")

    try:
        entry_point = entry_points[group.entry_point]
    except KeyError:
        entry_point = tomlkit.table()
        entry_points[group.entry_point] = entry_point

    if not isinstance(entry_point, Table):
        raise TypeError(f"{entry_point} is not a toml Table.")
    return entry_point


def _validate_plugin(plugin: Path) -> Path:
    py_proj = plugin.joinpath("pyproject.toml")
    if not py_proj.is_file():
        raise typer.BadParameter(f"Could not find file {py_proj}.")

    pkg_root = plugin.joinpath("src").joinpath(plugin.stem.lower())
    if not pkg_root.is_dir() or not pkg_root.joinpath("__init__.py").is_file():
        raise typer.BadParameter(
            f"{pkg_root} does not exist or is not a python package."
        )
    return plugin


def _validate_and_normalize_names(names: list[str]) -> set[str]:
    unique_normalized: set[str] = set()
    for name in names:
        validate_identifier(name)
        unique_normalized.add(name.lower())
    return unique_normalized


def _validate_package(package: str | None) -> str | None:
    if package is None:
        return None
    for pkg in package.split("."):
        validate_identifier(pkg)
    return package


def _validate_base(base: str | None, ctx: typer.Context) -> str | None:
    if base is None:
        return None

    group = Group(cast(str, ctx.params.get("group")))  # type: ignore[call-arg]
    for cls in group.bases:
        if cls.__name__ == base:
            return base
    raise typer.BadParameter(f"Invalid base {base} for group {group.value}.")


def _descend_and_create_as_needed(path: Path, packages: Sequence[str]) -> Path:
    if not packages:
        return path

    curr = path.joinpath(packages[0])
    if curr.exists() and not curr.is_dir():
        typer.echo(f"{curr} exists, but is not a directory.", file=sys.stderr)
        raise typer.Abort()

    curr.mkdir(exist_ok=True)
    if not curr.joinpath("__init__.py").is_file():
        curr.joinpath("__init__.py").touch()

    return _descend_and_create_as_needed(curr, packages[1:])


def _get_class(group: Group, base: str | None) -> type[Component]:
    if base is None:
        return group.bases[0]
    try:
        return next(cls for cls in group.bases if cls.__name__ == base)
    except StopIteration:
        raise RuntimeError(
            f"Invalid base {base} for group {group.value}. Please validate base"
            f"arg before calling _get_class."
        ) from None


def _to_cap_words(identifier: str) -> str:
    return "".join(w.capitalize() for w in identifier.split("_"))
