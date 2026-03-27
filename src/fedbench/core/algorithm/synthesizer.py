from abc import abstractmethod
from dataclasses import dataclass

from pandas import DataFrame

from fedbench.core.algorithm.context import (
    GlobalInitContext,
    SampleContext,
    TrainContext,
)
from fedbench.core.component import Component
from fedbench.core.payload import ArraysTarget, Payload


@dataclass(frozen=True)
class GlobalInitArtifacts:
    coordinator: Payload | None = None
    synthesizer: Payload | None = None


class Synthesizer(Component):
    """The framework view of the model to train and sample from."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def arrays_target(self) -> ArraysTarget:
        pass

    @property
    @abstractmethod
    def supports_coordinators(self) -> set[str]:
        pass

    @abstractmethod
    def global_init(
        self,
        dataset: DataFrame,
        context: GlobalInitContext,
    ) -> GlobalInitArtifacts:
        pass

    @abstractmethod
    def train(
        self,
        request: Payload,
        data: DataFrame,
        context: TrainContext,
    ) -> Payload:
        pass

    @abstractmethod
    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> DataFrame:
        pass
