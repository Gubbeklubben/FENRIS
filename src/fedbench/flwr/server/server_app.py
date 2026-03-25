from typing import cast

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fedbench.core.events import ClientsConfigured
from fedbench.core.payload import Payload
from fedbench.flwr.serde import FlwrSerde, Pickle
from fedbench.flwr.server.server import Strategy, configure_clients
from fedbench.runtime.component_factory import (
    create_centralized_eval_ctx,
    create_synthesizer,
)
from fedbench.runtime.early_stopping_monitor import EarlyStoppingMonitor
from fedbench.runtime.runcontext import RunContext


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()

    def _evaluate_fn(train_artifacts: Payload) -> float:
        # See also: pipeline.global_sample
        synthesizer = create_synthesizer(
            ctx.algorithm.synthesizer_spec.factory,
            artifacts=ctx.global_init_artifacts.synthesizer,
            client_cache=None,
        )
        num_synthetic_rows = (
            ctx.config.metrics.stop_synthetic_rows
            or ctx.config.num_synthetic_rows
            or ctx.dataset.global_holdout_size
        )
        synthetic_df = synthesizer.sample(
            train_artifacts, num_synthetic_rows, ctx.config.seed.sampling
        )

        eval_ctx = create_centralized_eval_ctx(ctx.config, ctx.dataset, synthetic_df)
        # noinspection PyUnnecessaryCast
        return ctx.eval_suite.global_evaluate_single(
            eval_ctx, cast(str, ctx.config.metrics.stop_metric)
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
        )
        state, metrics = strategy.run(grid, ctx.config.num_rounds)
        ctx.train_artifacts = state
        ctx.per_client_metrics = metrics

    return app
