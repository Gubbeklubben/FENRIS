import pytest
from flwr.server.strategy import FedAvg

from fedbench.common import MLRuntime
from fedbench.errors import DuplicateComponentError
from fedbench.server.registry import FlwrStrategyRegistry


def test_register_flwr_strategy() -> None:
    reg = FlwrStrategyRegistry(MLRuntime.NUMPY)
    @reg.flwr_strategy
    def factory() -> FedAvg:
        return FedAvg()

    assert reg._flwr_strategy_factory is factory


def test_can_not_register_flwr_strategy_twice() -> None:
    reg = FlwrStrategyRegistry(MLRuntime.NUMPY)
    @reg.flwr_strategy
    def some() -> FedAvg:
        return FedAvg()

    with pytest.raises(DuplicateComponentError):
        @reg.flwr_strategy
        def other() -> FedAvg:
            return FedAvg()

    assert reg._flwr_strategy_factory is some

