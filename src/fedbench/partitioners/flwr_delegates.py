from typing import Literal, cast

from datasets import Dataset  # type: ignore
from flwr_datasets.partitioner import (
    ContinuousPartitioner,
    DirichletPartitioner,
    ExponentialPartitioner,
    IidPartitioner,
    LinearPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
    SquarePartitioner,
)
from flwr_datasets.partitioner import Partitioner as FlwrPartitioner
from pandas import DataFrame

from fedbench.core.data.partitioner import Partitioner


class FlwrDelegatePartitioner(Partitioner):
    @classmethod
    def with_iid_partitioner(cls, num_partitions: int) -> Partitioner:
        return cls(IidPartitioner(num_partitions))

    @classmethod
    def with_linear_partitioner(cls, num_partitions: int) -> Partitioner:
        return cls(LinearPartitioner(num_partitions))

    @classmethod
    def with_square_partitioner(cls, num_partitions: int) -> Partitioner:
        return cls(SquarePartitioner(num_partitions))

    @classmethod
    def with_exponential_partitioner(cls, num_partitions: int) -> Partitioner:
        return cls(ExponentialPartitioner(num_partitions))

    @classmethod
    def with_dirichlet_partitioner(
        cls,
        num_partitions: int,
        partition_by: str,
        alpha: float,
        min_partition_size: int = 10,
        self_balancing: bool = False,
        seed: int | None = 42,
    ) -> Partitioner:
        return cls(
            DirichletPartitioner(
                num_partitions,
                partition_by=partition_by,
                alpha=alpha,
                min_partition_size=min_partition_size,
                self_balancing=self_balancing,
                shuffle=True,
                seed=seed,
            )
        )

    @classmethod
    def with_pathological_partitioner(
        cls,
        num_partitions: int,
        partition_by: str,
        num_classes_per_partition: int,
        class_assignment_mode: Literal[
            "random", "deterministic", "first-deterministic"
        ] = "random",
        seed: int | None = 42,
    ) -> Partitioner:
        return cls(
            PathologicalPartitioner(
                num_partitions,
                partition_by=partition_by,
                num_classes_per_partition=num_classes_per_partition,
                class_assignment_mode=class_assignment_mode,
                shuffle=True,
                seed=seed,
            )
        )

    @classmethod
    def with_shard_partitioner(
        cls,
        num_partitions: int,
        partition_by: str,
        num_shards_per_partition: int | None = None,
        shard_size: int | None = None,
        keep_incomplete_shard: bool = False,
        seed: int | None = 42,
    ) -> Partitioner:
        return cls(
            ShardPartitioner(
                num_partitions,
                partition_by=partition_by,
                num_shards_per_partition=num_shards_per_partition,
                shard_size=shard_size,
                keep_incomplete_shard=keep_incomplete_shard,
                shuffle=True,
                seed=seed,
            )
        )

    @classmethod
    def with_continuous_partitioner(
        cls,
        num_partitions: int,
        partition_by: str,
        strictness: float,
        seed: int | None = 42,
    ) -> Partitioner:
        return cls(
            ContinuousPartitioner(
                num_partitions,
                partition_by=partition_by,
                strictness=strictness,
                shuffle=True,
                seed=seed,
            )
        )

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
