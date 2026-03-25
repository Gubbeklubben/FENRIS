from fedbench.core.algorithm import Coordinator
from fedbench.runtime.registry import FactoryRegistry


def register_builtin_coordinators(registry: FactoryRegistry[Coordinator]) -> None:
    registry.add_builtin(
        "fedavg",
        f"{__package__}.fedavg:FedAvg",
    )
