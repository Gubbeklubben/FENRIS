from fedbench.algorithms.algorithm import Algorithm, Synthesizer, Aggregator
from fedbench.registry import FactoryRegistry


registry: FactoryRegistry[Algorithm] = FactoryRegistry(
    group=__package__,
    product_cls=Algorithm,
)
registry.add_builtin("fed_noop", f"{__package__}.fed_noop:FedNoop")


__all__ = [
    "registry",
    "Algorithm",
    "Synthesizer",
    "Aggregator"
]