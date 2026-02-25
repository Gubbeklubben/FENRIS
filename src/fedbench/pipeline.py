import json
from collections.abc import Iterable
from pathlib import Path

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


def federated_train_eval_loop(ctx: RunContext) -> None:
    from flwr.simulation import run_simulation
    from fedbench.flwr import client_app, make_server_app

    run_simulation(
        client_app=client_app,
        server_app=make_server_app(ctx),
        num_supernodes=ctx.config.num_clients,
    )


def global_sample(ctx: RunContext) -> None:
    synthesizer = ctx.components.algorithm.create_synthesizer()
    ctx.synthetic_df = synthesizer.sample(
        ctx.aggregated_state,
        ctx.config.num_synthetic_rows or 1,
        ctx.config.seed
    )


def global_evaluate(ctx: RunContext) -> None:
    # I imagine some of the metrics may be relevant here, but not all?
    # We can create a test set by concatenating all test sets from
    # partitioner.
    pass


def write_artifacts(ctx: RunContext) -> None:
    outputdir = Path(ctx.config.outputdir).joinpath(ctx.run_id)
    outputdir.mkdir(parents=True, exist_ok=False)

    with outputdir.joinpath("metrics.json").open("w") as f:
        json.dump(dict(ctx.aggregated_metrics), f)

    ctx.synthetic_df.to_csv(outputdir.joinpath("synthetic.csv"))


def default() -> Iterable[Command]:
    yield resolve_components
    yield try_loader
    yield federated_train_eval_loop
    yield global_sample
    yield global_evaluate
    yield write_artifacts