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
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def synthesizer_factory(self) -> Synthesizer:
        return FedAwesomeSynthesizer()

    def server_policy_factory(self) -> BaseServerPolicy:
        return FedAwesomeServerPolicy()


class FedAwesomeServerPolicy(FlwrStrategyDelegatePolicy):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @log_calls(__name__)
    def init(self, responses: Iterable[InitResponse]) -> ModelState:
        return [np.ndarray([1, 2, 3]), np.ndarray([3, 2, 1])]

    @log_calls(__name__)
    def flwr_strategy_factory(self) -> Strategy:
        return FedAvg()


class FedAwesomeSynthesizer(Synthesizer):
    def __init__(self):
        self._model_state = None

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @property
    def ml_runtime(self) -> MLRuntime:
        return MLRuntime.NUMPY

    @property
    def model_state(self) -> ModelState:
        return self._model_state

    @log_calls(__name__)
    def init(self, request: InitRequest) -> InitResponse:
        return InitResponse(
            request.client_id,
            {"whatever": np.ndarray([1, 2, 3])})

    def train(self, request: TrainRequest) -> TrainResponse:
        return TrainResponse(request.client_id, self._model_state, None, 1)

    def evaluate(self, request: EvalRequest) -> EvalResponse:
        return EvalResponse(request.client_id, None)

    def sample(self):
        raise NotImplementedError()