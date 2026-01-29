from abc import ABC, abstractmethod

from fedbench.common import MLRuntimeWeights
from fedbench.common import TrainPlan, TrainResult, EvalPlan, EvalResult


class Synthesizer(ABC):
    @abstractmethod
    def get_weights(self) -> MLRuntimeWeights:
        pass

    @abstractmethod
    def set_weights(self, weights: MLRuntimeWeights) -> None:
        pass

    # TODO! Probably also inject context information...
    @abstractmethod
    def train(self, plan: TrainPlan) -> TrainResult:
        pass

    @abstractmethod
    def evaluate(self, plan: EvalPlan) -> EvalResult | None:
        pass

    # TODO! Figure out signature...
    @abstractmethod
    def sample(self):
        pass

