import json
import math
from collections.abc import Iterable
from pathlib import Path

from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema as _infer_schema
from fedbench.core.eval import CentralizedEvalContext
from fedbench.core.logger import log_info
from fedbench.core.pipeline import Command
from fedbench.core.runcontext import RunContext
from fedbench.registries import (
    build_algorithm_registry,
    build_evaluator_registries,
    build_partitioner_registry,
)
from fedbench.resolver import (
    resolve_algorithm,
    resolve_df_loader,
    resolve_evaluators,
    resolve_partitioner,
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


def aggregate_per_client_metrics(ctx: RunContext) -> None:
    ctx.aggregated_metrics = ctx.eval_suite.aggregate(ctx.per_client_metrics.values())


def global_sample(ctx: RunContext) -> None:
    synthesizer = ctx.algorithm.create_synthesizer()
    ctx.synthetic_df = synthesizer.sample(
        ctx.aggregated_state,
        ctx.config.num_synthetic_rows or 1000,
        ctx.config.seed,
    )


def global_evaluate(ctx: RunContext) -> None:
    eval_ctx = CentralizedEvalContext(
        synthetic_df=ctx.synthetic_df,
        holdout_df=ctx.dataset.load_global_holdout(),
        client_train_df=ctx.dataset.load_all_train_data(),  # only used by MIA
        target_column=ctx.config.data.target_col,
        sensitive_columns=ctx.config.data.sensitive_cols,
        schema=ctx.dataset.schema,
        seed=ctx.config.seed,
    )
    ctx.centralized_metrics = ctx.eval_suite.global_evaluate(eval_ctx)
    pass


def write_artifacts(ctx: RunContext) -> None:
    outputdir = Path(ctx.config.outputdir).joinpath(ctx.run_id)
    outputdir.mkdir(parents=True, exist_ok=False)

    pairs = [
        ("federated", ctx.aggregated_metrics),
        ("centralized", ctx.centralized_metrics),
    ]

    for name, metrics in pairs:
        with outputdir.joinpath(f"metrics.{name}.json").open("w") as f:
            clean_metrics = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in metrics.items()
            }
            json.dump(clean_metrics, f, indent=4, allow_nan=False)

    ctx.synthetic_df.to_csv(outputdir.joinpath("synthetic.csv"), index=False)

    log_info(__file__, f"Benchmark artifacts written to {outputdir}.")


def pipeline() -> Iterable[Command]:
    yield resolve_components
    yield load_dataset
    yield federated_train_eval_loop
    yield aggregate_per_client_metrics
    yield global_sample
    yield global_evaluate
    yield write_artifacts
