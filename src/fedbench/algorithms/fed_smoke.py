from collections.abc import Iterable

from pandas import DataFrame

from fedbench.algorithms.algorithm import Algorithm
from fedbench.algorithms.synthesizer import Synthesizer
from fedbench.common import (
    MLRuntime,
    ModelState,
    InitRequest,
    InitResponse,
    TrainRequest,
    TrainResponse,
)
from fedbench.common import log_calls


class FedSmoke(Algorithm):
    @property
    def server_ml_runtime(self) -> MLRuntime:
        return MLRuntime.NUMPY

    def server_initialize(
            self,
            responses: Iterable[InitResponse]) -> ModelState:
        pass

    def server_aggregate(
            self,
            server_round: int,
            results: Iterable[TrainResponse]
    ) -> tuple[ModelState | None, dict[str, float] | None]:
        pass

    def synthesizer_factory(self) -> Synthesizer:
        return FedSmokeSynthesizer()


class FedSmokeSynthesizer(Synthesizer):
    @property
    def ml_runtime(self) -> MLRuntime:
        return MLRuntime.NUMPY

    @log_calls(__name__)
    def initialize(self, request: InitRequest) -> InitResponse:
        return super().initialize(request)

    @log_calls(__name__)
    def train(self, request: TrainRequest) -> TrainResponse:
        return request.create_response(request.model_state, None, 1)

    @log_calls(__name__)
    def sample(
            self,
            model_state: ModelState,
            num_rows: int,
            seed: int) -> DataFrame:
        return DataFrame()