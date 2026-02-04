from abc import ABC, abstractmethod
from collections.abc import Iterable
from socket import send_fds

from flwr.serverapp.strategy import Strategy

from fedbench.common import (
    MLRuntime,
    ModelState,
    TrainRequest,
    EvalRequest,
    TrainResponse,
    EvalResponse, InitResponse,
)


class BaseServerPolicy(ABC):
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def init(self, responses: Iterable[InitResponse]) -> ModelState:
        pass


class ServerPolicy(BaseServerPolicy):
    @property
    @abstractmethod
    def ml_runtime(self) -> MLRuntime:
        pass

    @abstractmethod
    def configure_train(
            self,
            server_round: int,
            model_state: ModelState,
            config: dict[str, bool | int | float | bytes],
            client_ids: Iterable[int]) -> Iterable[TrainRequest]:
        pass

    @abstractmethod
    def configure_evaluate(
            self,
            server_round: int,
            model_state: ModelState,
            config: dict[str, bool | int | float | bytes],
            client_ids: Iterable[int]) -> Iterable[EvalRequest]:
        pass

    @abstractmethod
    def aggregate_train(
            self,
            server_round: int,
            results: Iterable[TrainResponse]
    ) -> tuple[ModelState | None, dict[str, float] | None]:
        pass

    @abstractmethod
    def aggregate_evaluate(
            self,
            server_round: int,
            results: Iterable[EvalResponse]) -> dict[str, float] | None:
        pass


class NoopDefaultsPolicy(ServerPolicy, ABC):
    def configure_train(
            self,
            server_round: int,
            model_state: ModelState,
            config: dict[str, bool | int | float | bytes],
            client_ids: Iterable[int]) -> Iterable[TrainRequest]:

        return []

    def configure_evaluate(
            self,
            server_round: int,
            model_state: ModelState,
            config: dict[str, bool | int | float | bytes],
            client_ids: Iterable[int]) -> Iterable[EvalRequest]:

        return []

    def aggregate_evaluate(
            self,
            server_round: int,
            results: Iterable[EvalResponse]) -> dict[str, float] | None:

        return None


class FlwrStrategyDelegatePolicy(BaseServerPolicy):
    @abstractmethod
    def flwr_strategy_factory(self) -> Strategy:
        pass
