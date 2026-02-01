from typing import Annotated

import typer

from fedbench._import import validate_locator, load_registry
from fedbench._registry import ClientRegistry, ServerRegistry


app = typer.Typer()


def _validate_locator(locator: str) -> str:
    if not validate_locator(locator):
        raise typer.BadParameter(f"Invalid locator: {locator}")
    return locator


@app.command()
def run(
        client_reg: Annotated[str, typer.Option(callback=_validate_locator)],
        server_reg: Annotated[str, typer.Option(callback=_validate_locator)],
        num_nodes: int = typer.Option(default=3)) -> None:

    print(client_reg, server_reg, num_nodes)
    client = load_registry(client_reg)
    server = load_registry(server_reg)
    assert isinstance(client, ClientRegistry)
    assert isinstance(server, ServerRegistry)
    print(client, server)
    #run_simulation(server_app, client_app, 10)


if __name__ == "__main__":
    app()
