from abc import ABC, abstractmethod
from collections.abc import Iterable

from fedbench.common import (
    MLRuntimeWeights,
    TrainPlan,
    EvalPlan,
    TrainResult,
    EvalResult,
)


class ServerPolicy(ABC):
    @abstractmethod
    def configure_init(self):
        pass

    @abstractmethod
    def aggregate_init(self):
        pass

    @abstractmethod
    def configure_train(
            self,
            server_round: int,
            ml_runtime_weights: MLRuntimeWeights,
            config: dict[str, bool | int | float | bytes],
            node_ids: Iterable[int]) -> Iterable[TrainPlan]:
        pass

    @abstractmethod
    def configure_evaluate(
            self,
            server_round: int,
            ml_runtime_weights: MLRuntimeWeights,
            config: dict[str, bool | int | float | bytes],
            node_ids: Iterable[int]) -> Iterable[EvalPlan]:
        pass

    @abstractmethod
    def aggregate_train(
            self,
            server_round: int,
            results: Iterable[TrainResult]) -> TrainResult:
        pass

    @abstractmethod
    def aggregate_evaluate(
            self,
            server_round: int,
            results: Iterable[EvalResult]) -> EvalResult:
        pass
