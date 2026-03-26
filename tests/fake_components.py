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
from fedbench.runtime.registry import FactoryRegistry


class FakeSynthesizer(Synthesizer):
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


class FakeSynthRegistry(FactoryRegistry[Synthesizer]):
    KEY = "test_synthesizer"

    def __init__(self):
        super().__init__(f"{__package__}.synthesizers", Synthesizer)
        self._plugins = {
            self.KEY: FakeEntryPoint(
                self.KEY, "", f"{__package__}.synthesizers", FakeSynthesizer
            )
        }


class FakePartitionerRegistry(FactoryRegistry[Partitioner]):
    KEY = "test_partitioner"

    def __init__(self):
        super().__init__(f"{__package__}.partitioners", Partitioner)
        self._plugins = {
            self.KEY: FakeEntryPoint(
                self.KEY, "", f"{__package__}.partitioners", FakePartitioner
            )
        }
