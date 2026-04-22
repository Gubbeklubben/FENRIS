"""Defines the Coordinator and SingleStepCoordinator ABC's.

Classes
-------
Coordinator
    Abstract base class for Coordinator implementations.
SingleStepCoordinator
    Abstract Coordinator subclass with convenient methods for adapting common
    configure -> exec -> aggregate logic to the Coordinator base.
"""

from abc import abstractmethod
from collections.abc import Generator, Iterable

from fedbench.core.component import Component
from fedbench.core.payload import ArraysTarget, Payload


class Coordinator(Component):
    """Coordinator base class.

    Defines what the framework expects from any Coordinator implementation.
    Views one round of training as an arbitrary number of steps, each step
    consisting of a batch of requests, and associated replies.

    When one or more rounds of training has been completed, the framework expects
    to be able to call Coordinator.publish_train_artifacts and receive a
    `fedbench.core.payload.Payload`. This payload is used as input when
    sampling from the current synthesizer, expecting the end result to be
    a `pandas DataFrame` sampled with the current central model state.
    """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def arrays_target(self) -> ArraysTarget:
        """Array deserialization target.

        Decides the runtime type of the arrays field of
        `fedbench.core.payload.Payload` instances.

        Returns
        -------
        `fedbench.core.payload.ArraysTarget`
            numpy | torch
        """

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        """Attach preprocessing artifacts.

        Called just after __init__ if the preprocessing step produced
        relevant output.

        Parameters
        ----------
        artifacts : Payload
            The content of the coordinator field of the
            `fedbench.core.algorithm.synthesizer.GlobalInitArtifacts` instance
            resulting from calling global_init on the current synthesizer.
        """

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
        Iterable[tuple[int, `fedbench.core.payload.Payload`]]
            A batch of requests.

        Receives
        --------
        Iterable[tuple[int, `fedbench.core.payload.Payload`]]
            Replies to the previously yielded batch of requests.
        """

    @abstractmethod
    def publish_train_artifacts(self) -> Payload:
        """Publish training artifacts.

        Returns
        -------
        `fedbench.core.payload.Payload`
            Used as input to subsequent sampling from the current synthesizer.
        """


class SingleStepCoordinator(Coordinator):
    """Convenience adapter for common configure -> exec -> aggregate logic."""

    @abstractmethod
    def configure_train(
        self, client_ids: Iterable[int]
    ) -> Iterable[tuple[int, Payload]]:
        """Configure the next round of training.

        Parameters
        ----------
        client_ids : Iterable[int]
            All available clients.

        Returns
        -------
        Iterable[tuple[int, `fedbench.core.payload.Payload`]]
            Client ids and associated requests.
        """

    @abstractmethod
    def aggregate_train(self, replies: Iterable[tuple[int, Payload]]) -> None:
        """Aggregate training replies.

        Parameters
        ----------
        replies : Iterable[tuple[int, `fedbench.core.payload.Payload`]]
            Client ids and associated replies.
        """

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
