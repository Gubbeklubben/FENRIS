from abc import ABC, abstractmethod
from typing import Generator, Iterable

from fedbench.core.update import Update


class Coordinator(ABC):
    """Server side algorithm component."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def global_state(self) -> Update | None:
        """Current global state.

        The returned value, if not None, will be injected into the sampling function
        associated with the current Synthesizer.

        Returns
        -------
        Update | None
            A representation of the current global state.
        """

        return None

    def attach_global_init_artifacts(self, artifacts: Update) -> None:
        """Attach globally computed preprocessing artifacts.

        Called right after creating an instance if the global initialization step
        produced output, otherwise skipped.

        Parameters
        ----------
        artifacts : Update
            Output from calling the global_init function associated with the
            current Synthesizer.
        """

        pass

    @abstractmethod
    def train(
        self,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Update]],
        Iterable[tuple[int, Update]],
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


class SingleStepCoordinator(Coordinator):
    def configure_train(
        self, client_ids: Iterable[int]
    ) -> Iterable[tuple[int, Update]]:

        state = self.global_state
        if state is None:
            raise RuntimeError(
                f"{self}: No global state, can not use default 'configure_train'"
            )
        for cid in client_ids:
            yield cid, state

    @abstractmethod
    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        pass

    def train(
        self,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Update]],
        Iterable[tuple[int, Update]],
        None,
    ]:
        replies = yield self.configure_train(client_ids)
        self.aggregate_train(replies)
