from collections.abc import Callable

from fedbench.data.partitioners.partitioner import Partitioner
from fedbench.registry import Registry


def _factory_validator(
        factory: Callable[..., Partitioner]) -> Callable[..., Partitioner]:
    if not callable(factory):
        raise TypeError("Partitioner factory must be callable.")
    return factory


registry = Registry[Callable[..., Partitioner]](
    group=f"{__package__}",
    validator=_factory_validator,
)
registry.add_builtin(
    "iid-partitioner",
    f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_iid_partitioner"
)