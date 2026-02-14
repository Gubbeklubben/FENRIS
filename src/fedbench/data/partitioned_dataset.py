from typing import cast, Literal, Self

import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner, IidPartitioner


class PartitionedDataset:
    def __init__(
            self,
            df: pd.DataFrame,
            partitioner: Partitioner,
            test_size: float = 0.2,
            seed: int = 80085) -> None:

        self._partitioner: Partitioner = partitioner
        self._partitioner.dataset = Dataset.from_pandas(df)
        self._test_size = test_size
        self._seed = seed

    @classmethod
    def with_iid_partitioner(
            cls,
            df: pd.DataFrame,
            num_partitions: int,
            test_size: float = 0.2,
            seed: int = 80085) -> Self:

        return cls(
            df=df,
            partitioner=IidPartitioner(num_partitions),
            test_size=test_size,
            seed=seed)

    @property
    def num_partitions(self) -> int:
        # noinspection PyUnnecessaryCast
        return cast(int, self._partitioner.num_partitions)

    def load_partition(
            self,
            partition_id: int,
            split: Literal["train", "test"]) -> pd.DataFrame:

        return cast(
            pd.DataFrame,
            self._partitioner
                .load_partition(partition_id)
                .train_test_split(test_size=self._test_size, seed=self._seed)[split]
                .with_format("pandas")[:])

    def load_train_partition(self, partition_id: int) -> pd.DataFrame:
        return self.load_partition(partition_id, "train")

    def load_test_partition(self, partition_id: int) -> pd.DataFrame:
        return self.load_partition(partition_id, "test")