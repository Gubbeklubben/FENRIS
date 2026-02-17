from fedbench.algorithms.algorithm import Algorithm, Synthesizer, Aggregator
from fedbench.registry import Registry


def _alg_validator(value: type[Algorithm]) -> type[Algorithm]:
    if not issubclass(value, Algorithm):
        raise TypeError(
            f"The provided value {value} is not a subclass of {Algorithm}"
        )
    return value


registry: Registry[type[Algorithm]] = Registry(
    group=__package__,
    validator=_alg_validator
)
registry.add_builtin("fed_noop", f"{__package__}.fed_noop:FedNoop")


__all__ = [
    "registry",
    "Algorithm",
    "Synthesizer",
    "Aggregator"
]