from collections.abc import Callable
from enum import Enum

from flwr.serverapp.strategy import Strategy

from fedbench._registry import ServerRegistry
from fedbench.server.server_policy import ServerPolicy


_SERVER_POLICY_FACTORY = "_server_policy_factory"
_FLWR_STRATEGY_FACTORY = "_flwr_strategy_factory"


class _Components(Enum):
    CONF_INIT     = "_configure_init"
    AGGR_INIT     = "_aggregate_init"
    CONF_TRAIN    = "_configure_train"
    AGGR_TRAIN    = "_aggregate_train"
    CONF_EVALUATE = "_configure_evaluate"
    AGGR_EVALUATE = "_aggregate_evaluate"


class ServerPolicyRegistry(ServerRegistry):
    def server_policy(
            self,
            factory: Callable[[], ServerPolicy]) -> Callable[[], ServerPolicy]:

        return self._register(
            decorator_name="server_policy",
            attr_name=_SERVER_POLICY_FACTORY,
            plugin=factory)

    def get_server_policy_factory(self) -> Callable[[], ServerPolicy] | None:
        return getattr(self, _SERVER_POLICY_FACTORY, None)


class FlwrStrategyRegistry(ServerRegistry):
    def flwr_strategy(
            self, factory: Callable[[], Strategy]) -> Callable[[], Strategy]:

        return self._register(
            decorator_name="flwr_strategy",
            attr_name=_FLWR_STRATEGY_FACTORY,
            plugin=factory)

    def configure_init(self, func):
        raise NotImplementedError()

    def aggregate_init(self, func):
        raise NotImplementedError()

    def get_flwr_strategy_factory(self) -> Callable[[], Strategy] | None:
        return getattr(self, _FLWR_STRATEGY_FACTORY, None)

    def get_configure_init(self):
        raise NotImplementedError()

    def get_aggregate_init(self):
        raise NotImplementedError()


class ServerComponentRegistry(ServerRegistry):
    def configure_init(self, func):
        raise NotImplementedError()

    def aggregate_init(self, func):
        raise NotImplementedError()

    def configure_train(self, func):
        raise NotImplementedError()

    def aggregate_train(self, func):
        raise NotImplementedError()

    def configure_evaluate(self, func):
        raise NotImplementedError()

    def aggregate_evaluate(self, func):
        raise NotImplementedError()

    def get_configure_init(self):
        raise NotImplementedError()

    def get_aggregate_init(self):
        raise NotImplementedError()

    def get_configure_train(self):
        raise NotImplementedError()

    def get_aggregate_train(self):
        raise NotImplementedError()

    def get_configure_evaluate(self):
        raise NotImplementedError()

    def get_aggregate_evaluate(self):
        raise NotImplementedError()
