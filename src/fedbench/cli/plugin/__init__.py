import typer

from fedbench.cli.plugin.extend import app as add
from fedbench.cli.plugin.new import app as new

app = typer.Typer()
app.add_typer(new)
app.add_typer(add)
