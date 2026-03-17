from typing import Literal, cast

from datasets import Dataset  # type: ignore
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import Partitioner as FlwrPartitioner
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
        # Dataset.from_pandas preserves the DataFrame index as '__index_level_0__'
        # when the index is non-default (e.g. after train_test_split with shuffle=True).
        # Resetting the index before conversion prevents this column from leaking
        # into partition DataFrames and corrupting column selection downstream.
        self._flwr_partitioner.dataset = Dataset.from_pandas(df.reset_index(drop=True))

    def load_partition(
        self,
        partition_id: int,
        split: Literal["train", "test"],
        seed: int,
        test_size: float,
    ) -> DataFrame:

        # noinspection PyUnnecessaryCast
        return cast(
            DataFrame,
            self._flwr_partitioner.load_partition(partition_id)
            .train_test_split(test_size=test_size, seed=seed)[split]
            .to_pandas(),  # makes a copy; mutations will not affect underlying dataset
        )
