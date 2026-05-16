from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pandas import DataFrame

from fenris.core.component import Component


class Partitioner(Component):
    """Abstract base class for dataset partitioners.

    Partitioners divide a dataset into a fixed number of client partitions,
    each of which can be further split into train and test subsets on demand.
    """

    @property
    @abstractmethod
    def num_partitions(self) -> int:
        """Total number of client partitions."""

    @abstractmethod
    def set_dataset(self, df: DataFrame) -> None:
        """Load the full dataset for partitioning.

        Parameters
        ----------
        df : pandas.DataFrame
            The complete dataset to partition.
        """

    @abstractmethod
    def load_partition(
        self,
        partition_id: int,
        split: Literal["train", "test"],
        seed: int,
        test_size: float,
    ) -> DataFrame:
        """Return one split of a single client partition.

        Parameters
        ----------
        partition_id : int
            Zero-based index of the partition to load.
        split : {"train", "test"}
            Which split to return.
        seed : int
            Random seed for the train/test split.
        test_size : float
            Fraction of the partition reserved for the test split.

        Returns
        -------
        pandas.DataFrame
        """
