from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable

from fedbench.core.payload import ArraysTarget, Payload, PayloadSchema


class Coordinator(ABC):
    """Server side algorithm component."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def arrays_target(self) -> ArraysTarget:
        pass

    @property
    @abstractmethod
    def payload_schema(self) -> PayloadSchema:
        pass

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        """Attach globally computed preprocessing artifacts.

        Called right after creating an instance if the global initialization step
        produced output, otherwise skipped.

        Parameters
        ----------
        artifacts : Payload
            Output from calling the global_init function associated with the
            current Synthesizer.
        """

        pass

    @abstractmethod
    def train(
        self,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Payload]],
        Iterable[tuple[int, Payload]],
        None,
    ]:
        """Federated training.

        Can be divided into as many steps as desired. The framework consumes the
        provided generator by calling its send method to feed replies to yielded
        requests back into the generator.

        Parameters
        ----------
        client_ids : Iterable[int]
            All available clients.

        Yields
        ------
        Iterable[tuple[int, Update]]
            A batch of requests.

        Receives
        --------
        Iterable[tuple[int, Update]]
            Replies to the previously yielded batch of requests.
        """

        pass

    @abstractmethod
    def publish_train_artifacts(self) -> Payload:
        """Publish training artifacts.

        The returned Payload will be injected into the sample function associated
        with the current Synthesizer."""


class SingleStepCoordinator(Coordinator):
    @abstractmethod
    def configure_train(
        self, client_ids: Iterable[int]
    ) -> Iterable[tuple[int, Payload]]:
        pass

    @abstractmethod
    def aggregate_train(self, replies: Iterable[tuple[int, Payload]]) -> None:
        pass

    def train(
        self,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Payload]],
        Iterable[tuple[int, Payload]],
        None,
    ]:
        replies = yield self.configure_train(client_ids)
        self.aggregate_train(replies)
