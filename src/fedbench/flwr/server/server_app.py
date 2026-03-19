from flwr.common import Context
from flwr.server import Grid
from flwr.serverapp import ServerApp

from fedbench.core.events import ClientsConfigured
from fedbench.flwr.serde import FlwrSerde, Pickle
from fedbench.flwr.server.server import Strategy, configure_clients
from fedbench.runtime.component_factory import create_coordinator
from fedbench.runtime.runcontext import RunContext


def make_server_app(ctx: RunContext) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, _: Context) -> None:
        serde = FlwrSerde(
            object_serde=Pickle(disabled=ctx.config.disable_pickle),
            default_arrays_map=ctx.algorithm.coordinator_spec.arrays_to_ml_framework_map,
        )
        configure_clients(
            config=ctx.config,
            artifacts=ctx.global_init_artifacts.synthesizer,
            serde=serde,
            grid=grid,
        )
        ctx.eventbus.emit(ClientsConfigured())

        coordinator = create_coordinator(
            ctx.algorithm.coordinator_spec.factory,
            artifacts=ctx.global_init_artifacts.coordinator,
        )
        strategy = Strategy(
            seed=ctx.config.seed + 2,  # derived seed: generator init
            schema=ctx.dataset.schema,
            serde=serde,
            eventbus=ctx.eventbus,
            coordinator=coordinator,
        )
        state, metrics = strategy.run(grid, ctx.config.num_rounds)
        ctx.aggregated_state = state
        ctx.per_client_metrics = metrics

    return app
