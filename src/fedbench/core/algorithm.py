from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from dataclasses import dataclass

from pandas import DataFrame

from fedbench.core.data import TableSchema
from fedbench.core.types import Extras
from fedbench.core.update import Update


class Coordinator(ABC):
    """Server side algorithm component."""

    def __init__(
        self,
        config: Extras | None,
        artifacts: Update | None,
    ) -> None:
        self._config = config
        self._artifacts = artifacts

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def global_state(self) -> Update | None:
        return None

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


class Synthesizer(ABC):
    """The framework view of the model to train and sample from."""

    def __init__(
        self,
        config: Extras | None,
        artifacts: Update | None,
        client_cache: Update | None = None,
    ) -> None:
        self._config = config
        self._artifacts = artifacts
        self._client_cache = client_cache

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def fed_init(
        self,
        request: Update,
        seed: int,
        schema: TableSchema,
        data: DataFrame,
    ) -> Update:
        return Update()

    @abstractmethod
    def train(
        self,
        request: Update,
        data: DataFrame,
    ) -> Update:
        pass

    @abstractmethod
    def sample(
        self,
        request: Update,
        num_rows: int,
        seed: int,
    ) -> DataFrame:
        pass


@dataclass(frozen=True)
class ComponentSpec[T]:
    cls: type[T]  # Or a factory if we want to be more flexible...
    config: Extras | None = None
    arrays_to_ml_framework_map: dict[str, str] | None = None


def coordinator_spec(
    cls: type[Coordinator],
    config: Extras | None = None,
    arrays_to_ml_framework_map: dict[str, str] | None = None,
) -> ComponentSpec[Coordinator]:

    return ComponentSpec[Coordinator](cls, config, arrays_to_ml_framework_map)


def synthesizer_spec(
    cls: type[Synthesizer],
    config: Extras | None = None,
    arrays_to_ml_framework_map: dict[str, str] | None = None,
) -> ComponentSpec[Synthesizer]:

    return ComponentSpec[Synthesizer](cls, config, arrays_to_ml_framework_map)


def create_synthesizer(
    spec: ComponentSpec[Synthesizer],
    artifacts: Update | None,
    client_cache: Update | None,
) -> Synthesizer:

    instance = spec.cls(spec.config, artifacts, client_cache)
    if not isinstance(instance, Synthesizer):
        raise TypeError(f"{instance} is not a Synthesizer.")
    return instance


def create_coordinator(
    spec: ComponentSpec[Coordinator],
    artifacts: Update | None,
) -> Coordinator:

    instance =  spec.cls(spec.config, artifacts)
    if not isinstance(instance, Coordinator):
        raise TypeError(f"{instance} is not a Coordinator.")
    return instance


@dataclass(frozen=True)
class Artifacts:
    coordinator: Update | None = None
    synthesizer: Update | None = None


class Algorithm(ABC):
    """Algorithm entry point."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def coordinator_spec(self) -> ComponentSpec[Coordinator]:
        pass

    @property
    @abstractmethod
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        pass

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def global_init(
        self,
        seed: int,
        schema: TableSchema,
        dataset: DataFrame,
    ) -> Artifacts | None:
        """Do algorithm specific preprocessing on the full dataset."""

        return None