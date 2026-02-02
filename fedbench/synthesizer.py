from abc import ABC, abstractmethod

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


class Synthesizer(ABC):
    @property
    @abstractmethod
    def ml_runtime(self) -> MLRuntime:
        pass

    @property
    @abstractmethod
    def model_state(self) -> ModelState:
        pass

    @model_state.setter
    @abstractmethod
    def model_state(self, model_state: ModelState) -> None:
        pass

    @abstractmethod
    def init(self, request: InitRequest) -> InitResponse | None:
        pass

    @abstractmethod
    def train(self, request: TrainRequest) -> TrainResponse:
        pass

    @abstractmethod
    def evaluate(self, request: EvalRequest) -> EvalResponse | None:
        pass

    # TODO! Figure out signature...
    @abstractmethod
    def sample(self):
        pass

