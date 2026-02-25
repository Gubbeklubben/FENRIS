from fedbench.core.algorithm import Algorithm
from fedbench.core.factory_registry import FactoryRegistry


def register_builtin_algorithms(registry: FactoryRegistry[Algorithm]) -> None:
    registry.add_builtin("fed_noop", f"{__package__}.fed_noop:FedNoop")