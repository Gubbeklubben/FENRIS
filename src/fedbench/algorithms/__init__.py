from fedbench.algorithms.algorithm import Algorithm, Synthesizer, Aggregator
from fedbench.registry import Registry


def _validator(value: type[Algorithm]) -> type[Algorithm]:
    if not issubclass(value, Algorithm):
        raise TypeError(
            f"The provided value {value} is not a subclass of {Algorithm}"
        )
    return value


registry: Registry[type[Algorithm]] = Registry(__package__, _validator)
registry.add_builtin("fed_noop", f"{__package__}.fed_noop:FedNoop")


__all__ = [
    "registry",
    "Algorithm",
    "Synthesizer",
    "Aggregator"
]