import json
from collections.abc import Iterable
from pathlib import Path

from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema as _infer_schema
from fedbench.core.pipeline import Command
from fedbench.core.runcontext import RunContext
from fedbench.registries import (
    build_algorithm_registry,
    build_evaluator_registries,
    build_partitioner_registry,
)
from fedbench.resolver import (
    resolve_df_loader,
    resolve_algorithm,
    resolve_partitioner,
    resolve_evaluators,
)


def resolve_components(ctx: RunContext) -> None:
    ctx.df_loader = resolve_df_loader(ctx.config)

    ctx.algorithm = resolve_algorithm(
        ctx.config,
        build_algorithm_registry(),
    )
    ctx.partitioner = resolve_partitioner(
        ctx.config,
        build_partitioner_registry(),
    )
    ctx.eval_suite = resolve_evaluators(
        ctx.config,
        build_evaluator_registries(),
    )


def load_dataset(ctx: RunContext) -> None:
    df = ctx.df_loader()
    schema = _infer_schema(df)
    ctx.dataset = PartitionedDataset(
        df,
        schema,
        ctx.partitioner,
        ctx.config.test_size,
        ctx.config.seed,
    )


def federated_train_eval_loop(ctx: RunContext) -> None:
    from flwr.simulation import run_simulation
    from fedbench.flwr import client_app, make_server_app

    run_simulation(
        client_app=client_app,
        server_app=make_server_app(ctx),
        num_supernodes=ctx.config.num_clients,
    )


def global_sample(ctx: RunContext) -> None:
    synthesizer = ctx.algorithm.create_synthesizer()
    ctx.synthetic_df = synthesizer.sample(
        ctx.aggregated_state,
        ctx.config.num_synthetic_rows or 1,
        ctx.config.seed,
    )


def global_evaluate(ctx: RunContext) -> None:
    # Server's evaluation data can be retrieved using ctx.dataset.load_global_holdout()
    pass


def write_artifacts(ctx: RunContext) -> None:
    outputdir = Path(ctx.config.outputdir).joinpath(ctx.run_id)
    outputdir.mkdir(parents=True, exist_ok=False)

    with outputdir.joinpath("metrics.json").open("w") as f:
        json.dump(dict(ctx.aggregated_metrics), f)

    ctx.synthetic_df.to_csv(outputdir.joinpath("synthetic.csv"), index=False)


def pipeline() -> Iterable[Command]:
    yield resolve_components
    yield load_dataset
    yield federated_train_eval_loop
    yield global_sample
    yield global_evaluate
    yield write_artifacts
