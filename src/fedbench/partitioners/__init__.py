from fedbench.core.data import Partitioner
from fedbench.core.factory_registry import FactoryRegistry


def register_builtin_partitioners(registry: FactoryRegistry[Partitioner]) -> None:
    registry.add_builtin(
        "iid-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_iid_partitioner"
    )
