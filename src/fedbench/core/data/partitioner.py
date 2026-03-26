from abc import abstractmethod
from typing import Literal

from pandas import DataFrame

from fedbench.core.component import Component


class Partitioner(Component):
    @property
    @abstractmethod
    def num_partitions(self) -> int:
        pass

    @abstractmethod
    def set_dataset(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def load_partition(
        self,
        partition_id: int,
        split: Literal["train", "test"],
        seed: int,
        test_size: float,
    ) -> DataFrame:
        pass
