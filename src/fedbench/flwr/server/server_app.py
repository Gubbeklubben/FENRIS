from typing import cast

import pandas as pd
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fedbench.app.run.early_stopping_monitor import EarlyStoppingMonitor
from fedbench.app.run.runcontext import RunContext
from fedbench.core.algorithm import SampleContext
from fedbench.core.eval import CentralizedEvalContext
from fedbench.core.events import ClientsConfigured
from fedbench.core.logger import log_warning
from fedbench.flwr.serde import FlwrSerde, Pickle
from fedbench.flwr.server.server import Strategy, configure_clients


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()

    def _sample_synthetic_df(num_synthetic_rows: int) -> pd.DataFrame:
        # See also: pipeline.global_sample
        sample_ctx = SampleContext(
            global_init_artifacts=ctx.global_init_artifacts.synthesizer,
            client_storage=None,
            schema=ctx.dataset.schema,
            seed=ctx.config.seed.sampling,
            num_rows=num_synthetic_rows,
        )
        return ctx.synthesizer.sample(
            ctx.coordinator.publish_train_artifacts(), sample_ctx
        )

    def _evaluate_fn() -> float:
        num_synthetic_rows = (
            ctx.config.metrics.stop_synthetic_rows
            or ctx.config.num_synthetic_rows
            or ctx.dataset.global_holdout_size
        )

        eval_ctx = CentralizedEvalContext(
            synthetic_df=_sample_synthetic_df(num_synthetic_rows),
            holdout_df=ctx.dataset.load_global_holdout(),
            client_train_df=ctx.dataset.load_all_train_data(),
            target_column=ctx.config.data.target_col,
            sensitive_columns=ctx.config.data.sensitive_cols,
            schema=ctx.dataset.schema,
            seed=ctx.config.seed.evaluation,
        )
        # noinspection PyUnnecessaryCast
        return ctx.eval_suite.global_evaluate_single(
            eval_ctx, cast(str, ctx.config.metrics.stop_metric)
        )

    def _validate_fn() -> None:
        synthetic_df = _sample_synthetic_df(100)
        schema = ctx.dataset.schema

        # Check that no numeric column is all-NaN in D_syn
        for col in schema.numeric_columns(synthetic_df):
            if synthetic_df[col].isna().all():
                log_warning(
                    __name__,
                    f"Schema validation failed. "
                    f"Numeric column `{col}` is all-NaN in D_syn. "
                    f"Affected metric keys will be NaN.",
                )

        # Check that no nominal column has zero unique non-null values in D_syn
        for col in schema.nominal_columns(synthetic_df):
            if synthetic_df[col].nunique(dropna=True) == 0:
                log_warning(
                    __name__,
                    f"Schema validation failed. Categorical column `{col}` "
                    f"has no unique non-null values in D_syn. "
                    f"Affected metric keys will be NaN.",
                )

    @app.main()
    def main(grid: Grid, _: Context) -> None:
        serde = FlwrSerde(
            object_serde=Pickle(disabled=ctx.config.disable_pickle),
            default_arrays_target=ctx.coordinator.arrays_target,
        )
        configure_clients(
            config=ctx.config,
            artifacts=ctx.global_init_artifacts.synthesizer,
            serde=serde,
            grid=grid,
        )
        ctx.eventbus.emit(ClientsConfigured())

        strategy = Strategy.from_seed_config(
            seed_config=ctx.config.seed,
            schema=ctx.dataset.schema,
            serde=serde,
            eventbus=ctx.eventbus,
            coordinator=ctx.coordinator,
            monitor=EarlyStoppingMonitor(ctx.config.metrics, _evaluate_fn),
            validate_fn=_validate_fn,
        )
        train_artifacts, metrics = strategy.run(grid, ctx.config.num_rounds)
        ctx.train_artifacts = train_artifacts
        ctx.per_client_metrics = metrics

    return app
