from collections.abc import Callable

from fedbench.data.loaders import load_csv
from fedbench.data.partitioned_dataset import PartitionedDataset
from fedbench.data.partitioner import Partitioner
from fedbench.data.schemas import TableSchema, ColumnSchema
from fedbench.registry import Registry


def _factory_validator(
        factory: Callable[..., Partitioner]) -> Callable[..., Partitioner]:
    if not callable(factory):
        raise TypeError("Partitioner factory must be callable.")
    return factory


partitioner_registry = Registry[Callable[..., Partitioner]](
    group=f"{__package__}.partitioners",
    validator=_factory_validator,
)
partitioner_registry.add_builtin(
    "iid-partitioner",
    f"{__package__}.flwr_delegate_partitioner:FlwrDelegatePartitioner.with_iid_partitioner"
)


__all__ = [
    "load_csv",
    "partitioner_registry",
    "PartitionedDataset",
    "TableSchema",
    "ColumnSchema",
]