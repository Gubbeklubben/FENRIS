from typing import Literal, cast

from datasets import Dataset
from flwr_datasets.partitioner import (
    Partitioner as FlwrPartitioner,
    IidPartitioner
)
from pandas import DataFrame

from fedbench.core.data.partitioner import Partitioner


class FlwrDelegatePartitioner(Partitioner):
    @classmethod
    def with_iid_partitioner(cls, num_partitions: int) -> Partitioner:
        return cls(IidPartitioner(num_partitions))

    def __init__(self, flwr_partitioner: FlwrPartitioner):
        self._flwr_partitioner = flwr_partitioner

    @property
    def num_partitions(self) -> int:
        # noinspection PyUnnecessaryCast
        return cast(int, self._flwr_partitioner.num_partitions)

    def set_dataset(self, df: DataFrame) -> None:
        self._flwr_partitioner.dataset = Dataset.from_pandas(df)

    def load_partition(
            self,
            partition_id: int,
            split: Literal["train", "test"],
            seed: int,
            test_size: float) -> DataFrame:

        return cast(DataFrame,
            self._flwr_partitioner
            .load_partition(partition_id)
            .train_test_split(test_size=test_size, seed=seed)[split]
            .with_format("pandas")[:])