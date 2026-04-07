import http as whatever
import logging as blogging
import logging.handlers
import threading as reading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pandas import DataFrame

from fedbench.cli.subclass_builder import AbstractMethodCollector, Builder
from fedbench.core.algorithm import SingleStepCoordinator, Synthesizer
from fedbench.core.algorithm.context import (
    GlobalInitContext,
    SampleContext,
    TrainContext,
)
from fedbench.core.component import Component
from fedbench.core.payload import ArraysTarget, Payload


class DoNotVisitButPleaseReference(ABC):
    @abstractmethod
    def hell_no(self):
        pass


class MyWeirdMixin(ABC):
    @abstractmethod
    def mix_it_up(self) -> None:
        pass


@dataclass(frozen=True)
class DummyInitArtifacts:
    coordinator: Payload | None = None
    synthesizer: Payload | None = None


class Dummy(Component, MyWeirdMixin):
    """The framework view of the model to train and sample from."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def maybe(self) -> DoNotVisitButPleaseReference:
        pass

    @property
    @abstractmethod
    def arrays_target(self) -> ArraysTarget:
        pass

    @property
    @abstractmethod
    def supports_coordinators(self) -> set[str]:
        pass

    @abstractmethod
    def get_fed(self) -> reading.Thread:
        pass

    @abstractmethod
    def get_whatever(self) -> whatever.HTTPMethod:
        pass

    @abstractmethod
    def get_handler(self) -> blogging.handlers.QueueHandler:
        pass

    @abstractmethod
    def global_init(
        self,
        dataset: DataFrame,
        context: GlobalInitContext,
    ) -> DummyInitArtifacts:
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

    @abstractmethod
    def has_inner(self) -> bool:
        def inner():
            def why() -> None:
                pass

        return False


class DummySub(Dummy):
    @abstractmethod
    def other_arrays_target(self) -> ArraysTarget:
        pass


def main():
    targets = (DummySub, Synthesizer, SingleStepCoordinator)
    for target in targets:
        collector = AbstractMethodCollector(target)
        builder = Builder(collector)
        code = builder.build().code
        print(code)


if __name__ == "__main__":
    print(Dummy.mro())
    main()
