from fedbench.core.data import Partitioner
from fedbench.runtime.registry import FactoryRegistry


def register_builtin_partitioners(registry: FactoryRegistry[Partitioner]) -> None:
    registry.add_builtin(
        "iid-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_iid_partitioner",
    )
    registry.add_builtin(
        "linear-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_linear_partitioner",
    )
    registry.add_builtin(
        "square-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_square_partitioner",
    )
    registry.add_builtin(
        "exponential-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_exponential_partitioner",
    )
    registry.add_builtin(
        "dirichlet-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_dirichlet_partitioner",
    )
    registry.add_builtin(
        "pathological-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_pathological_partitioner",
    )
    registry.add_builtin(
        "shard-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_shard_partitioner",
    )
    registry.add_builtin(
        "continuous-partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_continuous_partitioner",
    )