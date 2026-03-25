from fedbench.core.data import Partitioner
from fedbench.runtime.registry import FactoryRegistry


def register_builtin_partitioners(registry: FactoryRegistry[Partitioner]) -> None:
    registry.add_builtin(
        "iid_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_iid_partitioner",
    )
    registry.add_builtin(
        "linear_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_linear_partitioner",
    )
    registry.add_builtin(
        "square_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_square_partitioner",
    )
    registry.add_builtin(
        "exponential_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_exponential_partitioner",
    )
    registry.add_builtin(
        "dirichlet_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_dirichlet_partitioner",
    )
    registry.add_builtin(
        "pathological_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_pathological_partitioner",
    )
    registry.add_builtin(
        "shard_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_shard_partitioner",
    )
    registry.add_builtin(
        "continuous_partitioner",
        f"{__package__}.flwr_delegates:FlwrDelegatePartitioner.with_continuous_partitioner",
    )
