from abc import ABC, abstractmethod
from typing import Iterable

from fedbench.common import TrainResult, EvalResult


class Aggregator(ABC):
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