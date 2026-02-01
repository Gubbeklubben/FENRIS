import importlib
import keyword
from typing import Annotated

import typer

from fedbench._registry import PluginRegistry


app = typer.Typer()


def _validate_locator(locator: str) -> str:
    module, _, attr = locator.partition(":")
    def valid(s):
        return s.isidentifier() and not keyword.iskeyword(s)

    if not all(valid(m) for m in module.split(".")) or not valid(attr):
        raise typer.BadParameter(f"Invalid locator: {locator}")

    return locator


def _load_registry(locator: str) -> PluginRegistry:
    module_name, _, attr = locator.partition(":")
    module = importlib.import_module(module_name)

    registry = getattr(module, attr)
    if not isinstance(registry, PluginRegistry):
        raise TypeError(f"Invalid registry type{type(registry)}")

    return registry


@app.command()
def run(
        client_reg: Annotated[str, typer.Option(callback=_validate_locator)],
        server_reg: Annotated[str, typer.Option(callback=_validate_locator)],
        num_nodes: int = typer.Option(default=3)) -> None:

    print(client_reg, server_reg, num_nodes)
    client = _load_registry(client_reg)
    server = _load_registry(server_reg)
    print(client, server)
    #run_simulation(server_app, client_app, 10)


if __name__ == "__main__":
    app()
