from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import fedbench.runtime.factory as factory
from fedbench.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
)
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import load_or_infer_schema
from fedbench.core.eval import CentralizedEvalContext, Evaluator
from fedbench.core.logger import log_info, log_warning
from fedbench.runtime.command import Command
from fedbench.runtime.platform_info import collect_platform_info
from fedbench.runtime.runcontext import RunContext


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


def global_init(ctx: RunContext) -> None:
    df = ctx.dataset.load_all_train_data()
    init_ctx = GlobalInitContext(
        schema=ctx.dataset.schema,
        seed=ctx.config.seed.init,
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
        schema=ctx.dataset.schema,
        seed=ctx.config.seed.sampling,
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
    with outputdir.joinpath("metadata.json").open("w") as f:
        json.dump(collect_platform_info(), f, indent=4, allow_nan=False)

    # Document which schema was used
    with outputdir.joinpath("schema.json").open("w") as f:
        json.dump(asdict(ctx.dataset.schema), f, indent=4, allow_nan=False)

    # Generate input schema file if requested by user
    if ctx.config.data.generate_input_schema:
        input_schema_path = Path(ctx.config.data.dataset).with_suffix(".schema.json")
        if input_schema_path.exists():
            # Config builder already checks this, but it doesn't hurt to double-check,
            # since some time could pass between the start and end of a run
            log_warning(
                __name__,
                f"Input schema file already exists "
                f"and will not be overwritten: {input_schema_path}",
            )
        shutil.copy(
            outputdir.joinpath("schema.json"),
            input_schema_path,
        )

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
