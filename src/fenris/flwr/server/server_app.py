from typing import cast

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fenris.app.run.early_stopping_monitor import EarlyStoppingMonitor
from fenris.app.run.runcontext import RunContext
from fenris.core.algorithm import SampleContext
from fenris.core.eval import CentralizedEvalContext
from fenris.core.events import ClientsConfigured
from fenris.core.payload import Payload
from fenris.flwr.serde import FlwrSerde, Pickle
from fenris.flwr.server.server import Strategy, configure_clients


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()

    def _evaluate_fn(train_artifacts: Payload) -> float:
        # See also: pipeline.global_sample
        num_synthetic_rows = (
            ctx.config.metrics.stop_synthetic_rows
            or ctx.config.num_synthetic_rows
            or ctx.dataset.global_holdout_size
        )
        sample_ctx = SampleContext(
            coordinator=ctx.coordinator.name,
            seed=ctx.config.seed.sampling,
            schema=ctx.dataset.schema,
            global_init_artifacts=ctx.global_init_artifacts.synthesizer,
            client_storage=None,
            num_rows=num_synthetic_rows,
        )
        synthetic_df = ctx.synthesizer.sample(train_artifacts, sample_ctx)

        eval_ctx = CentralizedEvalContext(
            synthetic_df=synthetic_df,
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

    @app.main()
    def main(grid: Grid, _: Context) -> None:
        serde = FlwrSerde(
            object_serde=Pickle(),
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
        )
        train_artifacts, metrics = strategy.run(grid, ctx.config.num_rounds)
        ctx.train_artifacts = train_artifacts
        ctx.per_client_metrics = metrics

    return app
