from flwr.common import Context
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench.component_factory import create_coordinator
from fedbench.core.events import ClientsConfigured
from fedbench.core.runcontext import RunContext
from fedbench.flwr.serde import (
    from_flwr_pickle,
    to_flwr_no_pickle,
    to_flwr_pickle,
)
from fedbench.flwr.server import send_config, send_artifacts, Strategy


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()

    to_flwr = to_flwr_no_pickle if ctx.config.disable_pickle else to_flwr_pickle
    from_flwr = from_flwr_pickle

    @app.main()
    def main(grid: Grid, _: Context) -> None:
        for reply in send_config(grid, ctx.config):
            if reply.has_error():
                raise RuntimeError(
                    f"Failed to send config: {reply.error.reason}"
                )

        for reply in send_artifacts(
                grid,
                to_flwr,
                ctx.global_init_artifacts.synthesizer
        ):
            if reply.has_error():
                raise RuntimeError(
                    f"Failed to send artifacts: {reply.error.reason}"
                )

        ctx.eventbus.emit(ClientsConfigured())

        coordinator = create_coordinator(
            spec=ctx.algorithm.coordinator_spec,
            artifacts=ctx.global_init_artifacts.coordinator,
        )
        arrays_map = ctx.algorithm.coordinator_spec.arrays_to_ml_framework_map

        strategy = Strategy(
            seed=ctx.config.seed,
            schema=ctx.dataset.schema,
            to_flwr=to_flwr,
            from_flwr=from_flwr,
            eventbus=ctx.eventbus,
            coordinator=coordinator,
            arrays_to_ml_framework_map=arrays_map
        )
        state, metrics = strategy.run(grid, ctx.config.num_rounds)
        ctx.aggregated_state = state
        ctx.per_client_metrics = metrics

    return app
