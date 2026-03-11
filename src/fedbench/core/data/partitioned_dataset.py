import pandas as pd
from sklearn.model_selection import train_test_split

from fedbench.core.data.partitioner import Partitioner
from fedbench.core.data.schemas import TableSchema


class PartitionedDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        schema: TableSchema,
        partitioner: Partitioner,
        test_size: float,
        seed: int,
    ) -> None:

        self._schema = schema
        self._partitioner = partitioner

        client_pool, holdout = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        self._partitioner.set_dataset(client_pool)
        self._global_holdout: pd.DataFrame = holdout

        self._test_size = test_size
        self._seed = seed

    @property
    def schema(self) -> TableSchema:
        return self._schema

    @property
    def num_partitions(self) -> int:
        return self._partitioner.num_partitions

    def load_global_holdout(self) -> pd.DataFrame:
        return self._global_holdout

    def load_train_partition(self, partition_id: int) -> pd.DataFrame:
        return self._partitioner.load_partition(
            partition_id=partition_id,
            split="train",
            test_size=self._test_size,
            seed=self._seed,
        )

    def load_test_partition(self, partition_id: int) -> pd.DataFrame:
        return self._partitioner.load_partition(
            partition_id=partition_id,
            split="test",
            test_size=self._test_size,
            seed=self._seed,
        )

    def load_all_train_data(self) -> pd.DataFrame:
        partitions = [
            self.load_train_partition(i)  # nofmt
            for i in range(self.num_partitions)
        ]
        return pd.concat(partitions, ignore_index=True)
