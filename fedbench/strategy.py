import numpy as np
from abc import ABC, abstractmethod
from typing import Protocol, Any

from fedbench.common import TrainPlan, TrainResult, EvalPlan, EvalResult


# Enable direct usage of Flower strategies with Flower backend
# without exposing Flower types in the core api.
class Strategy(Protocol):
    def configure_train(self, *args, **kwargs) -> Iterable[Any]:
        pass

    def aggregate_train(self, *args, **kwargs) -> tuple[Any | None]:
        pass

    def configure_evaluate(self, *args, **kwargs) -> Iterable[Any]:
        pass

    def aggregate_evaluate(self, *args, **kwargs) -> Any | None:
        pass


# Interface for those opting not to use Flower strategies. Will be adapted
# to Flower internally. (Could in some future be adapted to any suitable
# federation backend).
class FedBenchStrategy(ABC):
    @abstractmethod
    def configure_train(
            self,
            server_round: int,
            model_state: dict[str, np.ndarray],
            config: dict[str, bool | int | float | bytes ],
            node_ids: Iterable[int]) -> Iterable[TrainPlan]:
        pass

    @abstractmethod
    def aggregate_train(
            self,
            server_round: int,
            results: Iterable[TrainResult]) -> TrainResult:
        pass

    @abstractmethod
    def configure_evaluate(
            self,
            server_round: int,
            model_state: dict[str, np.ndarray],
            config: dict[str, bool | int | float | bytes ],
            node_ids: Iterable[int]) -> Iterable[EvalPlan]:
        pass

    @abstractmethod
    def aggregate_evaluate(
            self,
            server_round: int,
            results: Iterable[EvalResult]) -> EvalResult:
        pass