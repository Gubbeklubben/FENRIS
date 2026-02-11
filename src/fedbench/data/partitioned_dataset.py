from typing import Literal

import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner, IidPartitioner


class PartitionedDataset:

    def __init__(self,
        df: pd.DataFrame,
        partitioner: Partitioner,
        test_size: float = 0.2,
        seed: int = 80085
    ):
        self.partitioner = partitioner
        self.test_size = test_size
        self.seed = seed
        self.partitioner.dataset = Dataset.from_pandas(df)

    @classmethod
    def using_iid_partitioning(cls,
       df: pd.DataFrame,
       num_partitions: int,
       test_size: float = 0.2,
       seed: int = 80085
    ):
        return cls(
            df=df,
            partitioner=IidPartitioner(num_partitions),
            test_size=test_size,
            seed=seed
        )

    @property
    def num_partitions(self) -> int:
        return self.partitioner.num_partitions

    def load_partition(self, partition_id: int, split: Literal["train", "test"]) -> pd.DataFrame:
        return (
            self.partitioner
                .load_partition(partition_id)
                .train_test_split(test_size=self.test_size, seed=self.seed)[split]
                .with_format("pandas")[:]
        )

    def load_train_partition(self, partition_id: int) -> pd.DataFrame:
        return self.load_partition(partition_id, "train")

    def load_test_partition(self, partition_id: int) -> pd.DataFrame:
        return self.load_partition(partition_id, "test")