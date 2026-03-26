from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

from fedbench.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
)
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema as _infer_schema
from fedbench.core.logger import log_info
from fedbench.runtime.command import Command
from fedbench.runtime.component_factory import (
    create_centralized_eval_ctx,
    create_coordinator,
    create_df_loader,
    create_evaluation_suite,
    create_partitioner,
    create_synthesizer,
)
from fedbench.runtime.platform_info import collect_platform_info
from fedbench.runtime.registry_builder import (
    build_coordinator_registry,
    build_evaluator_registry,
    build_partitioner_registry,
    build_synthesizer_registry,
)
from fedbench.runtime.runcontext import RunContext


def create_components(ctx: RunContext) -> None:
    ctx.df_loader = create_df_loader(ctx.config)

    ctx.synthesizer = create_synthesizer(
        ctx.config,
        build_synthesizer_registry(),
    )
    ctx.coordinator = create_coordinator(
        ctx.config,
        build_coordinator_registry(),
    )
    ctx.partitioner = create_partitioner(
        ctx.config,
        build_partitioner_registry(),
    )
    ctx.eval_suite = create_evaluation_suite(
        ctx.config,
        build_evaluator_registry(),
    )


def load_dataset(ctx: RunContext) -> None:
    df = ctx.df_loader()
    schema = _infer_schema(df)
    ctx.dataset = PartitionedDataset(
        df,
        schema,
        ctx.partitioner,
        ctx.config.test_size,
        ctx.config.seed.partitioning,
    )


def global_init(ctx: RunContext) -> None:
    df = ctx.dataset.load_all_train_data()
    init_ctx = GlobalInitContext(
        schema=ctx.dataset.schema,
        seed=ctx.config.seed.init,
    )
    artifacts: GlobalInitArtifacts = ctx.synthesizer.global_init(df, init_ctx)
    ctx.global_init_artifacts = artifacts

    if artifacts.coordinator is not None:
        ctx.coordinator.attach_global_init_artifacts(artifacts.coordinator)


def federated_train_eval_loop(ctx: RunContext) -> None:
    from flwr.simulation import run_simulation

    from fedbench.flwr import client_app, make_server_app

    run_simulation(
        client_app=client_app,
        server_app=make_server_app(ctx),
        num_supernodes=ctx.config.num_clients,
    )


def aggregate_federated_metrics(ctx: RunContext) -> None:
    ctx.aggregated_metrics = {
        **ctx.eval_suite.aggregate(
            ctx.per_client_metrics.values(),
            ctx.config.data.target_col,
            ctx.config.data.sensitive_cols,
        ),
        **ctx.scalability_collector.get_metrics(),
    }


def global_sample(ctx: RunContext) -> None:
    sample_ctx = SampleContext(
        global_init_artifacts=ctx.global_init_artifacts.synthesizer,
        client_cache=None,
        seed=ctx.config.seed.sampling,
        num_rows=ctx.config.num_synthetic_rows
        or len(ctx.dataset.load_global_holdout()),
    )
    ctx.synthetic_df = ctx.synthesizer.sample(ctx.train_artifacts, sample_ctx)


def global_evaluate(ctx: RunContext) -> None:
    eval_ctx = create_centralized_eval_ctx(ctx.config, ctx.dataset, ctx.synthetic_df)
    ctx.centralized_metrics = ctx.eval_suite.global_evaluate(eval_ctx)


def write_artifacts(ctx: RunContext) -> None:
    outputdir = Path(ctx.config.outputdir).joinpath(ctx.run_id)
    outputdir.mkdir(parents=True, exist_ok=False)

    # Config snapshot
    with outputdir.joinpath("config_snapshot.json").open("w") as f:
        json.dump(ctx.config.jsondict(), f, indent=4)

    # Platform metadata
    with outputdir.joinpath("metadata.json").open("w") as f:
        json.dump(collect_platform_info(), f, indent=4, allow_nan=False)

    for name, metrics in [
        ("federated", ctx.aggregated_metrics),
        ("centralized", ctx.centralized_metrics),
    ]:
        clean = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in metrics.items()
        }
        with outputdir.joinpath(f"metrics.{name}.json").open("w") as f:
            json.dump(clean, f, indent=4, allow_nan=False)

    # Synthetic data
    ctx.synthetic_df.to_csv(outputdir.joinpath("synthetic.csv"), index=False)

    log_info(__name__, f"Benchmark artifacts written to {outputdir}.")


def pipeline() -> Iterable[Command]:
    yield create_components
    yield load_dataset
    yield global_init
    yield federated_train_eval_loop
    yield aggregate_federated_metrics
    yield global_sample
    yield global_evaluate
    yield write_artifacts
