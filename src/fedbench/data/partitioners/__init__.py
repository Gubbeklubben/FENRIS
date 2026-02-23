from fedbench.data.partitioners.partitioner import Partitioner
from fedbench.core.registry import FactoryRegistry


registry = FactoryRegistry[Partitioner](
    group=f"{__package__}",
    product_cls=Partitioner,  # type: ignore[type-abstract]
)

registry.add_builtin(
    "iid-partitioner",
    f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_iid_partitioner"
)