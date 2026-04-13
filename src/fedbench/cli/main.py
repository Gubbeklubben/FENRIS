import typer

from fedbench.cli.generate_schema import app as generate_schema
from fedbench.cli.plugin import app as plugins
from fedbench.cli.run import app as run
from fedbench.cli.show import app as show

app = typer.Typer()
app.add_typer(show)
app.add_typer(run)
app.add_typer(plugins, name="plugin")
app.add_typer(generate_schema)


if __name__ == "__main__":
    app()
