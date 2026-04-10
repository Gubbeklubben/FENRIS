from pathlib import Path
from typing import Annotated

import typer

from fedbench.cli.plugins._component import _Component

app = typer.Typer()


@app.command()
def add(
    components: Annotated[
        list[_Component],
        typer.Argument(
            parser=_Component.parse,
            metavar=f"{_Component.arg_syntax} ...",
            help="Generate entry point declarations and plugins "
            "implementations for supplied arguments. If no arguments, "
            "one Synthesizer is generated.",
        ),
    ],
    project_root: Annotated[
        Path | None, typer.Option(help="Project root directory.")
    ] = None,
) -> None:
    """Add component scaffolds to an existing plugin project."""

    root_dir = project_root if project_root else Path.cwd()
    root_dir = root_dir.resolve()
