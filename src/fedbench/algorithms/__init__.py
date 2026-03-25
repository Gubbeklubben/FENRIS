from fedbench.core.algorithm import Algorithm
from fedbench.runtime.registry import FactoryRegistry


def register_builtin_algorithms(registry: FactoryRegistry[Algorithm]) -> None:
    # Example algorithms for demonstration/testing purposes
    registry.add_builtin(
        "fed_hello",
        f"{__package__}.fed_hello:FedHello",
    )
    registry.add_builtin(
        "fed_random",
        f"{__package__}.fed_random:FedRandom",
    )
    registry.add_builtin(
        "fed_naughty",
        f"{__package__}.fed_naughty:FedNaughty",
    )

    # Implementations of existing federated synthetic data generation algorithms
    registry.add_builtin(
        "fed_tab_diff",
        f"{__package__}.fedtabdiff:FedTabDiff",
    )
    registry.add_builtin(
        "fed_tgan",
        f"{__package__}.fed_tgan:FedTGAN",
    )
