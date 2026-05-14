from abc import ABC
from typing import Literal, cast

from datasets import Dataset  # type: ignore[attr-defined]
from flwr_datasets.partitioner import Partitioner as FlwrPartitioner
from pandas import DataFrame

from fenris.core.data.partitioner import Partitioner


class _FlwrDelegatePartitioner(Partitioner, ABC):
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


class IidPartitioner(_FlwrDelegatePartitioner):
    def __init__(self, num_partitions: int) -> None:
        from flwr_datasets.partitioner import IidPartitioner as _IidPartitioner

        super().__init__(_IidPartitioner(num_partitions))


class LinearPartitioner(_FlwrDelegatePartitioner):
    def __init__(self, num_partitions: int) -> None:
        from flwr_datasets.partitioner import LinearPartitioner as _LinearPartitioner

        super().__init__(_LinearPartitioner(num_partitions))


class SquarePartitioner(_FlwrDelegatePartitioner):
    def __init__(self, num_partitions: int) -> None:
        from flwr_datasets.partitioner import SquarePartitioner as _SquarePartitioner

        super().__init__(_SquarePartitioner(num_partitions))


class ExponentialPartitioner(_FlwrDelegatePartitioner):
    def __init__(self, num_partitions: int) -> None:
        from flwr_datasets.partitioner import (
            ExponentialPartitioner as _ExponentialPartitioner,
        )

        super().__init__(_ExponentialPartitioner(num_partitions))


class DirichletPartitioner(_FlwrDelegatePartitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        alpha: float,
        seed: int,
        min_partition_size: int = 10,
        self_balancing: bool = False,
    ) -> None:
        from flwr_datasets.partitioner import (
            DirichletPartitioner as _DirichletPartitioner,
        )

        super().__init__(
            _DirichletPartitioner(
                num_partitions,
                partition_by=partition_by,
                alpha=alpha,
                min_partition_size=min_partition_size,
                self_balancing=self_balancing,
                shuffle=True,
                seed=seed,
            )
        )


class PathologicalPartitioner(_FlwrDelegatePartitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        num_classes_per_partition: int,
        seed: int,
        class_assignment_mode: Literal[
            "random", "deterministic", "first-deterministic"
        ] = "random",
    ) -> None:
        from flwr_datasets.partitioner import (
            PathologicalPartitioner as _PathologicalPartitioner,
        )

        super().__init__(
            _PathologicalPartitioner(
                num_partitions,
                partition_by=partition_by,
                num_classes_per_partition=num_classes_per_partition,
                class_assignment_mode=class_assignment_mode,
                shuffle=True,
                seed=seed,
            )
        )


class ShardPartitioner(_FlwrDelegatePartitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        seed: int,
        num_shards_per_partition: int | None = None,
        shard_size: int | None = None,
        keep_incomplete_shard: bool = False,
    ) -> None:
        from flwr_datasets.partitioner import ShardPartitioner as _ShardPartitioner

        super().__init__(
            _ShardPartitioner(
                num_partitions,
                partition_by=partition_by,
                num_shards_per_partition=num_shards_per_partition,
                shard_size=shard_size,
                keep_incomplete_shard=keep_incomplete_shard,
                shuffle=True,
                seed=seed,
            )
        )


class ContinuousPartitioner(_FlwrDelegatePartitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        strictness: float,
        seed: int,
    ) -> None:
        from flwr_datasets.partitioner import (
            ContinuousPartitioner as _ContinuousPartitioner,
        )

        super().__init__(
            _ContinuousPartitioner(
                num_partitions,
                partition_by=partition_by,
                strictness=strictness,
                shuffle=True,
                seed=seed,
            )
        )
