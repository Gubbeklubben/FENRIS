from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import fenris.app.factory as factory
from fenris.app.registry import Group
from fenris.app.run.command import Command
from fenris.app.run.partitioned_dataset import PartitionedDataset
from fenris.app.run.platform_info import collect_platform_info
from fenris.app.run.runcontext import RunContext
from fenris.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
)
from fenris.core.data.schemas import load_or_infer_schema
from fenris.core.eval import CentralizedEvalContext, Evaluator
from fenris.core.logger import log_info, log_warning
from fenris.core.payload import ArraysTarget


def create_components(ctx: RunContext) -> None:
    ctx.df_loader = factory.create_df_loader(ctx.config.data.dataset)

    ctx.partitioner = factory.create_partitioner(
        ctx.config.data.partitioner,
        ctx.config.data.partitioner_kwargs,
    )
    ctx.synthesizer = factory.create_synthesizer(
        ctx.config.synthesizer,
        ctx.config.synthesizer_kwargs,
    )
    ctx.coordinator = factory.create_coordinator(
        ctx.config.coordinator,
        ctx.config.coordinator_kwargs,
    )
    ctx.eval_suite = factory.create_evaluation_suite(ctx.config.metrics.run_categories)
    _validate_evaluators(ctx.eval_suite)


def _validate_evaluators(evaluators: Iterable[Evaluator]) -> None:
    existing_keys: dict[str, str] = {}

    for evaluator in evaluators:
        keys = list(evaluator.get_metric_keys())
        if not keys:
            raise ValueError(
                f"Evaluator {evaluator.name} does not declare any emitted metric keys."
            )
        for key in keys:
            if key in existing_keys:
                raise ValueError(
                    f"Metric key {key} is emitted multiple times, "
                    f"by both {evaluator.name} and {existing_keys[key]}."
                )
            existing_keys[key] = evaluator.name


def load_dataset(ctx: RunContext) -> None:
    df = ctx.df_loader()
    schema = load_or_infer_schema(Path(ctx.config.data.schema), df)
    ctx.dataset = PartitionedDataset(
        df, schema, ctx.partitioner, ctx.config.test_size, ctx.config.seed.partitioning
    )
    if ctx.dataset.num_dropped > 0:
        log_warning(
            __name__,
            f"Dropped {ctx.dataset.num_dropped} rows containing missing values before "
            f"partitioning ({len(df)} -> {len(df) - ctx.dataset.num_dropped} rows).",
        )


def global_init(ctx: RunContext) -> None:
    df = ctx.dataset.load_all_train_data()
    init_ctx = GlobalInitContext(
        coordinator=ctx.coordinator.name,
        seed=ctx.config.seed.init,
        schema=ctx.dataset.schema,
    )
    artifacts: GlobalInitArtifacts = ctx.synthesizer.global_init(df, init_ctx)
    ctx.global_init_artifacts = artifacts

    if not isinstance(artifacts, GlobalInitArtifacts):
        raise TypeError(
            f"Invalid value type returned from {ctx.synthesizer}.global_init(). "
            f"Expected: {GlobalInitArtifacts}. "
            f"Actual: {type(artifacts)}."
        )

    if artifacts.coordinator is not None:
        ctx.coordinator.attach_global_init_artifacts(artifacts.coordinator)


def federated_train_eval_loop(ctx: RunContext) -> None:
    from flwr.simulation import run_simulation

    from fenris.flwr import client_app, make_server_app

    num_cpus = ctx.config.client_cpus
    num_gpus = 0.0

    if (
        ctx.synthesizer.arrays_target == ArraysTarget.TORCH
        and os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""
    ):
        from torch import cuda

        if cuda.is_available():
            num_gpus = ctx.config.client_gpus

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    run_simulation(
        client_app=client_app,
        server_app=make_server_app(ctx),
        num_supernodes=ctx.config.num_clients,
        backend_config={
            "client_resources": {"num_cpus": num_cpus, "num_gpus": num_gpus}
        },
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
        coordinator=ctx.coordinator.name,
        seed=ctx.config.seed.sampling,
        schema=ctx.dataset.schema,
        global_init_artifacts=ctx.global_init_artifacts.synthesizer,
        client_storage=None,
        num_rows=ctx.config.num_synthetic_rows or ctx.dataset.global_holdout_size,
    )
    ctx.synthetic_df = ctx.synthesizer.sample(ctx.train_artifacts, sample_ctx)


def global_evaluate(ctx: RunContext) -> None:
    eval_ctx = CentralizedEvalContext(
        synthetic_df=ctx.synthetic_df,
        holdout_df=ctx.dataset.load_global_holdout(),
        client_train_df=ctx.dataset.load_all_train_data(),
        target_column=ctx.config.data.target_col,
        sensitive_columns=ctx.config.data.sensitive_cols,
        schema=ctx.dataset.schema,
        seed=ctx.config.seed.evaluation,
    )
    ctx.centralized_metrics = ctx.eval_suite.global_evaluate(eval_ctx)


def write_artifacts(ctx: RunContext) -> None:
    outputdir = Path(ctx.config.outputdir).joinpath(ctx.run_id)
    outputdir.mkdir(parents=True, exist_ok=False)

    # Config snapshot
    with outputdir.joinpath("config_snapshot.json").open("w") as f:
        json.dump(ctx.config.jsondict(), f, indent=4)

    # Platform metadata
    with outputdir.joinpath("platform_info.json").open("w") as f:
        json.dump(collect_platform_info(), f, indent=4, allow_nan=False)

    # Document which schema was used
    with outputdir.joinpath("schema.json").open("w") as f:
        json.dump(asdict(ctx.dataset.schema), f, indent=4, allow_nan=False)

    component_meta = {}
    for comp in ctx.components:
        reg = Group.from_type(type(comp)).get_registry()
        component_meta[comp.name] = asdict(reg.get_metadata(comp.name))

    with outputdir.joinpath("components.json").open("w") as f:
        json.dump(component_meta, f, indent=4)

    # Synthetic data
    ctx.synthetic_df.to_csv(outputdir.joinpath("synthetic.csv"), index=False)

    # Metrics
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
