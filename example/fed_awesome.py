from typing import Iterable

from flwr.serverapp.strategy import Strategy, FedAvg

from fedbench.algorithm import Algorithm
from fedbench.common import ModelState, InitResponse
from fedbench.server_policy import BaseServerPolicy, FlwrStrategyDelegatePolicy
from fedbench.synthesizer import Synthesizer


class FedAwesome(Algorithm):
    def synthesizer_factory(self) -> Synthesizer:
        raise NotImplementedError()

    def server_policy_factory(self) -> BaseServerPolicy:
        return FedAwesomeServerPolicy()


class FedAwesomeServerPolicy(FlwrStrategyDelegatePolicy):
    def init(self, responses: Iterable[InitResponse]) -> ModelState:
        return []

    def flwr_strategy_factory(self) -> Strategy:
        return FedAvg()
