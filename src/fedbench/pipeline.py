from collections.abc import Iterable

from fedbench.core.pipeline import Command
from fedbench.core.runcontext import RunContext
from fedbench.registries import (
    build_algorithm_registry,
    build_partitioner_registry,
    build_evaluator_registries
)
from fedbench.resolver import resolve_components as _resolve_components


def resolve_components(ctx: RunContext) -> None:
    components = _resolve_components(
        ctx.config,
        build_algorithm_registry(),
        build_partitioner_registry(),
        build_evaluator_registries()
    )
    ctx.components = components


def try_loader(ctx: RunContext) -> None:
    # Crash early if loader fails
    ctx.components.df_loader()


def run_federation(ctx: RunContext) -> None:
    from flwr.simulation import run_simulation
    from fedbench.flwr import client_app, make_server_app

    run_simulation(
        client_app=client_app,
        server_app=make_server_app(ctx),
        num_supernodes=ctx.config.num_clients,
    )


def default() -> Iterable[Command]:
    yield resolve_components
    yield try_loader
    yield run_federation