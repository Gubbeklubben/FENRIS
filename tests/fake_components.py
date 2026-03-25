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


class FakeAlgRegistry(FactoryRegistry[Algorithm]):
    KEY = "test_algorithm"

    def __init__(self):
        super().__init__(f"{__package__}.algorithms", Algorithm)
        self._plugins = {
            self.KEY: FakeEntryPoint(
                self.KEY, "", f"{__package__}.algorithms", FakeAlgorithm
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
