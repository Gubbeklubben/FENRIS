from abc import ABC, abstractmethod
from collections.abc import Iterable

from pandas import DataFrame

from fedbench.common import (
    MLRuntime,
    ModelState,
    InitRequest,
    InitResponse,
    TrainRequest,
    TrainResponse,
)


class Synthesizer(ABC):
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def ml_runtime(self) -> MLRuntime:
        pass

    # noinspection PyMethodMayBeStatic
    def initialize(self, request: InitRequest) -> InitResponse:
        return request.create_response(None)

    @abstractmethod
    def train(self, request: TrainRequest) -> TrainResponse:
        pass

    @abstractmethod
    def sample(self, model_state: ModelState, num_rows: int, seed: int) -> DataFrame:
        pass

