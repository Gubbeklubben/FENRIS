from pandas import DataFrame

from fedbench.data.schemas import TableSchema
from fedbench.data.partitioners.partitioner import Partitioner


class PartitionedDataset:
    def __init__(
            self,
            df: DataFrame,
            schema: TableSchema,
            partitioner: Partitioner,
            test_size: float,
            seed: int) -> None:

        self._schema = schema
        self._partitioner = partitioner
        self._partitioner.set_dataset(df)
        self._test_size = test_size
        self._seed = seed

    @property
    def schema(self) -> TableSchema:
        return self._schema

    @property
    def num_partitions(self) -> int:
        return self._partitioner.num_partitions

    def load_train_partition(self, partition_id: int) -> DataFrame:
        return self._partitioner.load_partition(
            partition_id=partition_id,
            split="train",
            test_size=self._test_size,
            seed=self._seed)

    def load_test_partition(self, partition_id: int) -> DataFrame:
        return self._partitioner.load_partition(
            partition_id=partition_id,
            split="test",
            test_size=self._test_size,
            seed=self._seed
        )