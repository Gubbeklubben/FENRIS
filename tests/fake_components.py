from types import SimpleNamespace
from typing import Generator, Iterable, Literal

import pytest
from pandas import DataFrame

from fenris.app.registry import Registry
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


def _mock_entry_points(group: str) -> list[SimpleNamespace]:
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


@pytest.fixture
def synthesizers(monkeypatch):
    monkeypatch.setattr("fenris.app.registry.entry_points", _mock_entry_points)
    return Registry(group="fenris.synthesizers")


@pytest.fixture
def coordinators(monkeypatch):
    monkeypatch.setattr("fenris.app.registry.entry_points", _mock_entry_points)
    return Registry(group="fenris.coordinators")


@pytest.fixture
def partitioners(monkeypatch):
    monkeypatch.setattr("fenris.app.registry.entry_points", _mock_entry_points)
    return Registry(group="fenris.partitioners")
