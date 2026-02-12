from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from pandas import DataFrame

from fedbench.common import Arrays, MessageContent


class ServerComponent(ABC):
    """Server side synthesizer component.

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
            client_ids: Iterable[int]) -> Iterable[MessageContent]:
        return ()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def aggregate_init(
            self,
            replies: Iterable[MessageContent]
    ) -> tuple[Arrays | None, dict[str, Any] | None]:
        return None, None

    @abstractmethod
    def aggregate_train(
            self,
            replies: Iterable[MessageContent]
    ) -> tuple[Arrays | None, dict[str, Any] | None]:
        pass


class ClientComponent(ABC):
    """Client side synthesizer component.

    Instances live and die inside one training or evaluation round.
    """
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    def arrays_decode_spec(self) -> dict[str, str] | None:
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def init(self, request: MessageContent) -> MessageContent:
        return MessageContent()

    @abstractmethod
    def train(
            self,
            request: MessageContent,
            data: DataFrame) -> MessageContent:
        pass

    @abstractmethod
    def sample(
            self,
            request: MessageContent,
            num_rows: int,
            seed: int) -> DataFrame:
        pass


class Synthesizer(ABC):
    @property
    def non_array_protocols(self) -> tuple[str, ...]:
        return ()

    @property
    @abstractmethod
    def server_factory(self) -> ServerComponent:
        pass

    @abstractmethod
    def client_factory(self) -> ClientComponent:
        pass