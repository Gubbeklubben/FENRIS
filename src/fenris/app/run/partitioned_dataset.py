from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from fenris.core.data.partitioner import Partitioner
from fenris.core.data.schemas import TableSchema


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

        df_clean = df.dropna().reset_index(drop=True)
        self.num_dropped = len(df) - len(df_clean)
        df = df_clean

        from sklearn.model_selection import train_test_split

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

    @property
    def global_holdout_size(self) -> int:
        return len(self._global_holdout)

    def load_global_holdout(self) -> pd.DataFrame:
        return self._global_holdout.copy()

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
        partitions = [self.load_train_partition(i) for i in range(self.num_partitions)]
        import pandas as _pd

        return _pd.concat(partitions, ignore_index=True)
