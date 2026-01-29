from abc import ABC, abstractmethod
from typing import Iterable

from fedbench.common import TrainPlan, EvalPlan, MLRuntimeWeights


class Configurator(ABC):
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