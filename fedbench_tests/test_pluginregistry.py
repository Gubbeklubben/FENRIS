import pytest
from flwr.server.strategy import FedAvg

# noinspection PyProtectedMember
from fedbench._pluginregistry import _PluginRegistry, PluginRegistryException
from fedbench.common import MLRuntime


def test_register_flwr_strategy() -> None:
    instance = _PluginRegistry()
    @instance.register_flwr_strategy(ml_runtime=MLRuntime.NUMPY)
    def factory() -> FedAvg:
        return FedAvg()

    assert instance.flwr_strategy_factory is factory
    assert instance.server_ml_runtime == MLRuntime.NUMPY
    assert instance.has_flower_strategy_factory()


def test_can_not_register_flwr_strategy_twice() -> None:
    instance = _PluginRegistry()
    @instance.register_flwr_strategy(ml_runtime=MLRuntime.NUMPY)
    def some() -> FedAvg:
        return FedAvg()

    with pytest.raises(PluginRegistryException):
        @instance.register_flwr_strategy(ml_runtime=MLRuntime.NUMPY)
        def other() -> FedAvg:
            return FedAvg()

    assert instance.flwr_strategy_factory is some

