from abc import ABC, abstractmethod
from typing import Generator, Iterable

from fedbench.core.data import TableSchema
from fedbench.core.update import Update


class Coordinator(ABC):
    """Server side algorithm component."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def global_state(self) -> Update | None:
        return None

    def attach_global_init_artifacts(self, artifacts: Update) -> None:
        """Attach globally computed preprocessing artifacts.

        Override if you depend on global_init to do preprocessing. Always
        called right after creating an instance.
        """

        pass

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Update]],
        Iterable[tuple[int, Update]],
        None,
    ]:
        _ = yield ()

    @abstractmethod
    def train(
        self,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Update]],
        Iterable[tuple[int, Update]],
        None,
    ]:
        pass


class SingleStepCoordinator(Coordinator):
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def configure_fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Iterable[tuple[int, Update]]:
        return ()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        return None

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

    def fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Update]],
        Iterable[tuple[int, Update]],
        None,
    ]:
        replies = yield self.configure_fed_init(seed, schema, client_ids)
        self.aggregate_fed_init(replies)

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
