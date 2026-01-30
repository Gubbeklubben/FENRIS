from typing import Callable

from flwr.serverapp.strategy import Strategy

from fedbench._registry import PluginRegistry
from fedbench.server.server_policy import ServerPolicy

# python >= 3.12
type FlwrStrategyFactory = Callable[[], Strategy]
type ServerPolicyFactory = Callable[[], ServerPolicy]

_SERVER_POLICY_KEY = "ServerPolicy"
_FLWR_STRATEGY_KEY = "FlowerStrategy"
_CONF_INIT_KEY = "configure_init"
_AGGR_INIT_KEY = "aggregate_init"
_CONF_TRAIN_KEY = "configure_train"
_AGGR_TRAIN_KEY = "aggregate_train"
_CONF_EVALUATE_KEY = "configure_evaluate"
_AGGR_EVALUATE_KEY = "aggregate_evaluate"


class ServerPolicyRegistry(PluginRegistry):
    def server_policy(self, factory: ServerPolicyFactory) -> ServerPolicyFactory:
        return self._register(_SERVER_POLICY_KEY, factory)

    # Internal ComponentResolver api
    @property
    def _server_policy_factory(self) -> ServerPolicyFactory:
        return self._get(_SERVER_POLICY_KEY)


class FlwrStrategyRegistry(PluginRegistry):
    def flwr_strategy(self, factory: FlwrStrategyFactory) -> FlwrStrategyFactory:
        return self._register(_FLWR_STRATEGY_KEY, factory)

    def configure_init(self, func):
        return self._register(_CONF_INIT_KEY, func)

    def aggregate_init(self, func):
        return self._register(_AGGR_INIT_KEY, func)

    # Internal ComponentResolver api
    @property
    def _flwr_strategy_factory(self) -> FlwrStrategyFactory:
        return self._get(_FLWR_STRATEGY_KEY)

    @property
    def _configure_init(self):
        return self._get(_CONF_INIT_KEY)

    @property
    def _aggregate_init(self):
        return self._get(_AGGR_INIT_KEY)


class ServerComponentRegistry(PluginRegistry):
    def configure_init(self, func):
        return self._register(_CONF_INIT_KEY, func)

    def aggregate_init(self, func):
        return self._register(_AGGR_INIT_KEY, func)

    def configure_train(self, func):
        return self._register(_CONF_TRAIN_KEY, func)

    def aggregate_train(self, func):
        return self._register(_AGGR_TRAIN_KEY, func)

    def configure_evaluate(self, func):
        return self._register(_CONF_EVALUATE_KEY, func)

    def aggregate_evaluate(self, func):
        return self._register(_AGGR_EVALUATE_KEY, func)

    # Internal ComponentResolver api
    @property
    def _configure_init(self):
        return self._get(_CONF_INIT_KEY)

    @property
    def _aggregate_init(self):
        return self._get(_AGGR_INIT_KEY)

    @property
    def _configure_train(self):
        return self._get(_CONF_TRAIN_KEY)

    @property
    def _aggregate_train(self):
        return self._get(_AGGR_TRAIN_KEY)

    @property
    def _configure_evaluate(self):
        return self._get(_CONF_EVALUATE_KEY)

    @property
    def _aggregate_evaluate(self):
        return self._get(_AGGR_EVALUATE_KEY)
