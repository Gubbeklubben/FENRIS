from typing import Literal

from pandas import DataFrame

from fedbench.core.algorithm import Algorithm, ComponentSpec, Synthesizer
from fedbench.core.data import Partitioner
from fedbench.core.payload import Payload
from fedbench.runtime.registry import FactoryRegistry


class FakeSynthesizer(Synthesizer):
    def train(self, request: Payload, data: DataFrame) -> Payload:
        return NotImplemented

    def sample(self, request: Payload, num_rows: int, seed: int) -> DataFrame:
        return NotImplemented


class FakeAlgorithm(Algorithm):
    @property
    def supports_coordinators(self) -> set[str]:
        return set()

    @property
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        return NotImplemented


class FakePartitioner(Partitioner):
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
    def __init__(self, name, value, group):
        self.name = name
        self.value = value
        self.group = group
        self.product = None

    def load(self):
        return self.product


class FakeAlgRegistry(FactoryRegistry[Algorithm]):
    def __init__(self):
        super().__init__(f"{__package__}.algorithms", Algorithm)
        self._plugins = {
            "test": FakeEntryPoint("test", "test", f"{__package__}.algorithms")
        }
        self._plugins["test"].product = FakeAlgorithm


class FakePartitionerRegistry(FactoryRegistry[Partitioner]):
    def __init__(self):
        super().__init__(f"{__package__}.partitioners", Partitioner)
        self._plugins = {
            "test": FakeEntryPoint("test", "test", f"{__package__}.partitioners")
        }
        self._plugins["test"].product = FakePartitioner
