import typer

from fedbench.cli.plugins import app as plugins
from fedbench.cli.run import app as run
from fedbench.cli.show import app as show

app = typer.Typer()
app.add_typer(show)
app.add_typer(run)
app.add_typer(plugins, name="plugins")


if __name__ == "__main__":
    app()
