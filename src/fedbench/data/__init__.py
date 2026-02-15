from collections.abc import Callable
from fedbench.data.partitioned_dataset import PartitionedDataset
from fedbench.registry import Registry


# registry: Registry[Callable[[], PartitionedDataset]] = Registry("dataset",
# ...)