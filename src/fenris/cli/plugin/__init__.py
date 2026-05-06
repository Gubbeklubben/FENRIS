import typer

from fenris.cli.plugin.extend import app as add
from fenris.cli.plugin.new import app as new

app = typer.Typer()
app.add_typer(new)
app.add_typer(add)
