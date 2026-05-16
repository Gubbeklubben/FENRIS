from __future__ import annotations

import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import Annotated

import tomlkit
import typer
from tomlkit.toml_document import TOMLDocument

from fenris.cli.plugin._util import validate_identifier

app = typer.Typer()


@app.command()
def new(
    name: Annotated[
        str,
        typer.Argument(
            help="Project name.",
            callback=validate_identifier,
        ),
    ],
    parent_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Create a new plugin project.

    A plugin project can supply any number of components to any available
    entry point group.
    """
    parent = parent_dir if parent_dir is not None else Path.cwd()
    parent = parent.resolve()

    root = parent.joinpath(name)
    if root.exists():
        typer.echo(f"{root} already exists.", file=sys.stderr)
        raise typer.Abort()

    root.mkdir(parents=True)
    with root.joinpath("pyproject.toml").open("w") as f:
        tomlkit.dump(_create_toml(name.lower()), f)

    root_pkg = root.joinpath("src").joinpath(name.lower())
    root_pkg.mkdir(parents=True)
    root_pkg.joinpath("__init__.py").touch()


def _create_toml(project_name: str) -> TOMLDocument:
    toml = tomlkit.document()
    toml.add(
        tomlkit.comment(f"Created by {__name__} using tomlkit {tomlkit.__version__}.")
    )
    project = tomlkit.table()
    project.add("name", project_name)
    project.add("version", "0.1.0")
    project.add("description", "")
    project.add("authors", tomlkit.array())
    project.add("license", "MIT")
    project.add(tomlkit.nl())
    dist = distribution(__name__.split(".")[0])
    deps = tomlkit.array()
    deps.append(f"{dist.name} (>={dist.version},<={dist.version})")
    project.add("dependencies", deps)
    toml.add("project", project)
    return toml
