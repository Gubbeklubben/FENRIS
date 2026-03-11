from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from pandas import DataFrame

from fedbench.core.algorithm.coordinator import Coordinator
from fedbench.core.algorithm.synthesizer import Synthesizer
from fedbench.core.data import TableSchema
from fedbench.core.update import Update


@dataclass(frozen=True)
class ComponentSpec[T]:
    factory: Callable[[], T]
    arrays_to_ml_framework_map: dict[str, str] | None = None


def coordinator_spec(
    factory: Callable[[], Coordinator],
    arrays_to_ml_framework_map: dict[str, str] | None = None,
) -> ComponentSpec[Coordinator]:

    return ComponentSpec[Coordinator](factory, arrays_to_ml_framework_map)


def synthesizer_spec(
    factory: Callable[[], Synthesizer],
    arrays_to_ml_framework_map: dict[str, str] | None = None,
) -> ComponentSpec[Synthesizer]:

    return ComponentSpec[Synthesizer](factory, arrays_to_ml_framework_map)


@dataclass(frozen=True)
class GlobalInitArtifacts:
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
    ) -> GlobalInitArtifacts | None:
        """Do algorithm specific preprocessing on the full dataset.

        Resulting artifacts are injected into Coordinator/Synthesizer instances
        via the 'attach_global_init_artifacts' method. Don't be tempted to
        stash them on self and capture in a factory, that is not intended
        to work.
        """

        return None