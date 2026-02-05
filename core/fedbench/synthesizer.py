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
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def ml_runtime(self) -> MLRuntime:
        pass

    @abstractmethod
    def init(self, request: InitRequest) -> InitResponse:
        pass

    @abstractmethod
    def train(self, request: TrainRequest) -> TrainResponse:
        pass

    # As per meeting 04.02. If the metrics computations are to be
    # implemented client side as a first, I am thinking this method
    # should be entirely under our control, and dropped from the public
    # interface. It is all not 100% clear to me yet.
    @abstractmethod
    def evaluate(self, request: EvalRequest) -> EvalResponse:
        pass

    # TODO! Figure out signature, this is the method for generating data.
    @abstractmethod
    def sample(self):
        pass

