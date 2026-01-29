from typing import Callable

from flwr.serverapp.strategy import Strategy

from fedbench.aggregator import Aggregator
from fedbench.common import MLRuntime
from fedbench.configurator import Configurator
from fedbench.synthesizer import Synthesizer

type SynthesizerFactory = Callable[[], Synthesizer]
type ServerComponentFactory = Callable[[], tuple[Aggregator, Configurator | None]]
type FlwrStrategyFactory = Callable[[], Strategy]


class PluginRegistryException(Exception):
    pass


class _PluginRegistry:
    def __init__(self):
        self._synthesizer_factory = None
        self._synthesizer_ml_runtime = None
        self._server_component_factory = None
        self._flwr_strategy_factory = None
        self._server_ml_runtime = None
        self._metric_fns = {}

    def register_synthesizer(
            self, ml_runtime: MLRuntime
    ) -> Callable[[SynthesizerFactory], SynthesizerFactory]:
        """Register a synthesizer factory."""

        if self._synthesizer_factory is not None:
            raise PluginRegistryException(
                "Synthesizer factory already registered")

        def decorator(factory: SynthesizerFactory) -> SynthesizerFactory:
            self._synthesizer_factory = factory
            self._synthesizer_ml_runtime = ml_runtime
            return factory
        return decorator

    def register_server_components(
            self, ml_runtime: MLRuntime
    ) -> Callable[[ServerComponentFactory], ServerComponentFactory]:
        """Register a server component factory."""

        if self._server_component_factory is not None:
            raise PluginRegistryException("Server component factory already registered")

        if self._flwr_strategy_factory is not None:
            raise PluginRegistryException("Flower Strategy factory already registered")

        def decorator(factory: ServerComponentFactory) -> ServerComponentFactory:
            self._server_component_factory = factory
            self._server_ml_runtime = ml_runtime
            return factory
        return decorator

    def register_flwr_strategy(
            self, ml_runtime: MLRuntime
    ) -> Callable[[FlwrStrategyFactory], FlwrStrategyFactory]:
        """Register a flwr strategy factory."""

        if self._server_component_factory is not None:
            raise PluginRegistryException("Server component factory already registered")

        if self._flwr_strategy_factory is not None:
            raise PluginRegistryException("Flower Strategy factory already registered")

        def decorator(factory: FlwrStrategyFactory) -> FlwrStrategyFactory:
            self._flwr_strategy_factory = factory
            self._server_ml_runtime = ml_runtime
            return factory
        return decorator

    def has_flower_strategy_factory(self) -> bool:
        return self._flwr_strategy_factory is not None

    @property
    def synthesizer_factory(self) -> SynthesizerFactory:
        return self._synthesizer_factory

    @property
    def synthesizer_ml_runtime(self) -> MLRuntime:
        return self._synthesizer_ml_runtime

    @property
    def server_component_factory(self) -> ServerComponentFactory:
        return self._server_component_factory

    @property
    def flwr_strategy_factory(self) -> FlwrStrategyFactory:
        return self._flwr_strategy_factory

    @property
    def server_ml_runtime(self) -> MLRuntime:
        return self._server_ml_runtime


instance = _PluginRegistry()