from typing import Iterable

import numpy as np
from flwr.serverapp.strategy import Strategy, FedAvg

from fedbench.algorithm import Algorithm
from fedbench.common import (
    MLRuntime,
    ModelState,
    InitRequest,
    InitResponse,
    TrainRequest,
    TrainResponse,
    EvalRequest,
    EvalResponse,
)
from fedbench.common import log_calls
from fedbench.server_policy import BaseServerPolicy, FlwrStrategyDelegatePolicy
from fedbench.synthesizer import Synthesizer


class FedAwesome(Algorithm):
    def synthesizer_factory(self) -> Synthesizer:
        return FedAwesomeSynthesizer()

    def server_policy_factory(self) -> BaseServerPolicy:
        return FedAwesomeServerPolicy()


class FedAwesomeServerPolicy(FlwrStrategyDelegatePolicy):
    @log_calls(__name__)
    def init(self, responses: Iterable[InitResponse]) -> ModelState:
        return [np.array([1, 2, 3]), np.array([3, 2, 1])]

    @log_calls(__name__)
    def flwr_strategy_factory(self) -> Strategy:
        return FedAvg()


class FedAwesomeSynthesizer(Synthesizer):
    @property
    def ml_runtime(self) -> MLRuntime:
        return MLRuntime.NUMPY

    @property
    def model_state(self) -> ModelState:
        return self._model_state

    @log_calls(__name__)
    def init(self, request: InitRequest) -> InitResponse:
        return request.create_response({"whatever": np.array([1, 2, 3])})

    @log_calls(__name__)
    def train(self, request: TrainRequest) -> TrainResponse:
        return request.create_response(request.model_state, None, 1)

    def evaluate(self, request: EvalRequest) -> EvalResponse:
        return request.create_response(None)

    def sample(self):
        raise NotImplementedError()