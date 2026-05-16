from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from types import SimpleNamespace
from typing import Literal

from pandas import DataFrame

from fenris.core.algorithm import (
    Coordinator,
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fenris.core.data import Partitioner
from fenris.core.payload import ArraysTarget, Payload


class FakeSynthesizer(Synthesizer):
    SUPPORTED_COORDINATORS = {"fake_coordinator"}

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    def global_init(
        self, df: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:
        pass

    def train(self, request: Payload, df: DataFrame, context: TrainContext) -> Payload:
        pass

    def sample(self, request: Payload, context: SampleContext) -> DataFrame:
        pass


class FakeCoordinator(Coordinator):
    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    def train(
        self, client_ids: Iterable[int]
    ) -> Generator[
        Iterable[tuple[int, Payload]],
        Iterable[tuple[int, Payload]],
        None,
    ]:
        pass

    def publish_train_artifacts(self) -> Payload:
        pass


class FakePartitioner(Partitioner):
    # noinspection PyUnusedLocal
    def __init__(self, num_partitions: int):
        pass

    @property
    def num_partitions(self) -> int:
        return 0

    def set_dataset(self, df: DataFrame) -> None:
        return NotImplemented

    def load_partition(
        self,
        partition_id: int,
        split: Literal["train", "test"],
        seed: int,
        test_size: float,
    ) -> DataFrame:
        return NotImplemented


class NotASynthesizer:
    SUPPORTED_COORDINATORS = {"fake_coordinator"}

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    def global_init(
        self, df: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:
        pass

    def train(self, request: Payload, df: DataFrame, context: TrainContext) -> Payload:
        pass

    def sample(self, request: Payload, context: SampleContext) -> DataFrame:
        pass


def not_a_synth_entry_points(group: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name="not_a_synthesizer",
            group="fenris.synthesizers",
            value="fake:NotASynthesizer",
            module="fake",
            attr="NotASynthesizer",
            dist=SimpleNamespace(name="", version=""),
            load=lambda: NotASynthesizer,
        )
    ]


class AbstractSynthesizer(Synthesizer, ABC):
    SUPPORTED_COORDINATORS = {"fake_coordinator"}

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    def global_init(
        self, df: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:
        pass

    def train(self, request: Payload, df: DataFrame, context: TrainContext) -> Payload:
        pass

    def sample(self, request: Payload, context: SampleContext) -> DataFrame:
        pass

    @abstractmethod
    def abstract(self) -> None:
        pass


def abstract_synth_entry_points(group: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name="abstract_synthesizer",
            group="fenris.synthesizers",
            value="fake:AbstractSynthesizer",
            module="fake",
            attr="AbstractSynthesizer",
            dist=SimpleNamespace(name="", version=""),
            load=lambda: AbstractSynthesizer,
        )
    ]


def not_a_class_entry_points(group: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name="not_a_class",
            group="fenris.synthesizers",
            value="fake:NOT_A_CLASS",
            module="fake",
            attr="NOT_A_CLASS",
            dist=SimpleNamespace(name="", version=""),
            load=lambda: object(),
        )
    ]


def sane_entry_points(group: str) -> list[SimpleNamespace]:
    match group:
        case "fenris.synthesizers":
            ep = SimpleNamespace(
                name="fake_synthesizer",
                group="fenris.synthesizers",
                value="fake:FakeSynth",
                module="fake",
                attr="FakeSynth",
                dist=SimpleNamespace(name="", version=""),
                load=lambda: FakeSynthesizer,
            )
        case "fenris.coordinators":
            ep = SimpleNamespace(
                name="fake_coordinator",
                group="fenris.coordinators",
                value="fake:FakeCoord",
                module="fake",
                attr="FakeCoord",
                dist=SimpleNamespace(name="", version=""),
                load=lambda: FakeCoordinator,
            )
        case "fenris.partitioners":
            ep = SimpleNamespace(
                name="fake_partitioner",
                group="fenris.partitioner",
                value="fake:FakePartitioner",
                module="fake",
                attr="FakePartitioner",
                dist=SimpleNamespace(name="", version=""),
                load=lambda: FakePartitioner,
            )
        case _:
            raise ValueError(group)
    return [ep]
