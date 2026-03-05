from fedbench.core.algorithm import Algorithm
from fedbench.core.factory_registry import FactoryRegistry


def register_builtin_algorithms(registry: FactoryRegistry[Algorithm]) -> None:
    registry.add_builtin(
        "fed_hello", f"{__package__}.fed_hello:FedHello",
    )
    registry.add_builtin(
        "fed_tab_diff", f"{__package__}.fedtabdiff:FedTabDiff",
    )
    registry.add_builtin(
        "fed_random", f"{__package__}.fed_random:FedRandom",
    )