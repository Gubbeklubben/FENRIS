from abc import ABC, abstractmethod
from collections.abc import Iterable, Generator

from pandas import DataFrame

from fedbench.core.data import TableSchema
from fedbench.core.update import Update


class Coordinator(ABC):
    """Server side algorithm component."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return None

    # We could choose to return global state from aggregate hooks.
    # However, the contract is: The returned Update contains the
    # current global state, one way or the other, such that the framework
    # can feed it into Synthesizer.sample and expect it to sample
    # using whatever the current global state is.
    # Therefore, I prefer to let the coordinator keep this state,
    # and ask for it when needed.
    @property
    def global_state(self) -> Update | None:
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def fed_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int],) -> Generator[Iterable[tuple[int, Update]],
                                                    Iterable[tuple[int, Update]],
                                                    None,]:
        _ = yield ()

    @abstractmethod
    def train(
            self,
            client_ids: Iterable[int],) -> Generator[Iterable[tuple[int, Update]],
                                                    Iterable[tuple[int, Update]],
                                                    None,]:
        pass


class SingleStepCoordinator(Coordinator):
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def configure_fed_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int],) -> Iterable[tuple[int, Update]]:
        return ()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def aggregate_fed_init(
            self,
            replies: Iterable[tuple[int, Update]]) -> None:
        return None

    def configure_train(
            self,
            client_ids: Iterable[int]) -> Iterable[tuple[int, Update]]:

        state = self.global_state
        if state is None:
            raise RuntimeError(
                f"{self}: No global state, can not use default 'configure_train'"
            )
        for cid in client_ids:
            yield cid, state

    @abstractmethod
    def aggregate_train(
            self,
            replies: Iterable[tuple[int, Update]]) -> None:
        pass

    def fed_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int],) -> Generator[Iterable[tuple[int, Update]],
                                                    Iterable[tuple[int, Update]],
                                                    None,]:
        replies = yield self.configure_fed_init(seed, schema, client_ids)
        self.aggregate_fed_init(replies)

    def train(
            self,
            client_ids: Iterable[int],) -> Generator[Iterable[tuple[int, Update]],
                                                    Iterable[tuple[int, Update]],
                                                    None,]:
        replies = yield self.configure_train(client_ids)
        self.aggregate_train(replies)


class Synthesizer(ABC):
    """The framework view of the model to train and sample from."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def fed_init(
            self,
            request: Update,
            seed: int,
            schema: TableSchema,
            data: DataFrame,) -> Update:
        return Update()

    @abstractmethod
    def train(
            self,
            request: Update,
            data: DataFrame,) -> Update:
        pass

    @abstractmethod
    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int,) -> DataFrame:
        pass


class Algorithm(ABC):
    """Algorithm entry point."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def create_coordinator(self) -> Coordinator:
        pass

    @abstractmethod
    def create_synthesizer(self) -> Synthesizer:
        pass