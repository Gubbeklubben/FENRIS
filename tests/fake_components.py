from typing import Literal

from pandas import DataFrame

from fedbench.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fedbench.core.data import Partitioner
from fedbench.core.payload import ArraysTarget, Payload
from fedbench.runtime.registry import Registry


class FakeSynthesizer(Synthesizer):
    @property
    def name(self) -> str:
        return "fake_synthesizer"

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    @property
    def supports_coordinators(self) -> set[str]:
        return set()

    def global_init(
        self, dataset: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:
        pass

    def train(
        self, request: Payload, data: DataFrame, context: TrainContext
    ) -> Payload:
        pass

    def sample(self, request: Payload, context: SampleContext) -> DataFrame:
        pass


class FakePartitioner(Partitioner):
    @property
    def name(self) -> str:
        return "fake_partitioner"

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


class FakeEntryPoint:
    def __init__(self, name, value, group, product):
        self.name = name
        self.value = value
        self.group = group
        self.product = product

    def load(self):
        return self.product


class FakeSynthRegistry(Registry):
    KEY = "fake_synthesizer"

    def __init__(self):
        super().__init__(f"{__package__}.fake_synthesizers")
        self._entry_points = {
            self.KEY: FakeEntryPoint(
                self.KEY, "", f"{__package__}.fake_synthesizers", FakeSynthesizer
            )
        }


class FakePartitionerRegistry(Registry):
    KEY = "fake_partitioner"

    def __init__(self):
        super().__init__(f"{__package__}.fake_partitioners")
        self._entry_points = {
            self.KEY: FakeEntryPoint(
                self.KEY, "", f"{__package__}.fake_partitioners", FakePartitioner
            )
        }
