import typer

from fedbench.cli.plugins.add import app as add
from fedbench.cli.plugins.new import app as new

app = typer.Typer()
app.add_typer(new)
app.add_typer(add)
