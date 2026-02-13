from abc import ABC, abstractmethod
from collections.abc import Iterable

from pandas import DataFrame

from fedbench.common import Update


class Aggregator(ABC):
    """Server side algorithm component.

    An instance lives for one entire simulation.
    """
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    def arrays_decode_spec(self) -> dict[str, str] | None:
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def configure_init(
            self,
            client_ids: Iterable[int]) -> Iterable[tuple[int, Update]]:
        return ()

    @abstractmethod
    def aggregate_init(
            self,
            replies: Iterable[Update]) -> Update:
        pass

    @abstractmethod
    def aggregate_train(
            self,
            replies: Iterable[Update]) -> Update:
        pass


class Synthesizer(ABC):
    """Client side algorithm component.

    Instances live and die inside one training or evaluation round.
    """
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    def arrays_decode_spec(self) -> dict[str, str] | None:
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def init(self, request: Update) -> Update:
        return Update()

    @abstractmethod
    def train(
            self,
            request: Update,
            data: DataFrame) -> Update:
        pass

    @abstractmethod
    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int) -> DataFrame:
        pass


class Algorithm(ABC):
    # TODO! Should be allowed to return a collection of acceptable choices
    @staticmethod
    def requires_non_array_protocol() -> str | None:
        return None

    @staticmethod
    @abstractmethod
    def aggregator_factory() -> Aggregator:
        pass

    @staticmethod
    @abstractmethod
    def synthesizer_factory() -> Synthesizer:
        pass