from importlib.metadata import version

import typer

from fenris.cli.generate_schema import app as generate_schema
from fenris.cli.plugin import app as plugins
from fenris.cli.run import app as run
from fenris.cli.show import app as show

app = typer.Typer(
    context_settings={
        "help_option_names": [
            "-h",
            "--help",
            "--fenris-help",  # Alias for profiling. PyCharm's profiler eats --help
        ]
    }
)
app.add_typer(show)
app.add_typer(run)
app.add_typer(plugins, name="plugin")
app.add_typer(generate_schema)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(version("fenris"))
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


if __name__ == "__main__":
    app()
