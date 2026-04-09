import typer

from fedbench.cli.run import app as run
from fedbench.cli.show import app as show

app = typer.Typer()
app.add_typer(show)
app.add_typer(run)


if __name__ == "__main__":
    app()
