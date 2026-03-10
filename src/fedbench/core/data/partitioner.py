from abc import ABC, abstractmethod
from typing import Literal

from pandas import DataFrame


class Partitioner(ABC):
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
