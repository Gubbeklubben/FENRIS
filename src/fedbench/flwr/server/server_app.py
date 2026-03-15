from flwr.common import Context
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench.core.events import ClientsConfigured
from fedbench.flwr.serde import FlwrSerde, Pickle
from fedbench.flwr.server.server import Strategy, send_artifacts, send_config
from fedbench.runtime.component_factory import create_coordinator
from fedbench.runtime.runcontext import RunContext


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()
    serde = FlwrSerde(
        object_serde=Pickle(ctx.config.disable_pickle),
        default_arrays_map=ctx.algorithm.coordinator_spec.arrays_to_ml_framework_map,
    )

    @app.main()
    def main(grid: Grid, _: Context) -> None:
        for reply in send_config(grid, ctx.config):
            if reply.has_error():
                raise RuntimeError(f"Failed to send config: {reply.error.reason}")

        for reply in send_artifacts(grid, serde, ctx.global_init_artifacts.synthesizer):
            if reply.has_error():
                raise RuntimeError(f"Failed to send artifacts: {reply.error.reason}")

        ctx.eventbus.emit(ClientsConfigured())

        coordinator = create_coordinator(
            ctx.algorithm.coordinator_spec.factory,
            artifacts=ctx.global_init_artifacts.coordinator,
        )

        strategy = Strategy(
            seed=ctx.config.seed,
            schema=ctx.dataset.schema,
            serde=serde,
            eventbus=ctx.eventbus,
            coordinator=coordinator,
        )
        state, metrics = strategy.run(grid, ctx.config.num_rounds)
        ctx.aggregated_state = state
        ctx.per_client_metrics = metrics

    return app
