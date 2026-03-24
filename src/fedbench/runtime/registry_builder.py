from fedbench.algorithms import register_builtin_algorithms
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import Partitioner
from fedbench.core.eval import Evaluator
from fedbench.evaluators import register_builtin_evaluators
from fedbench.partitioners import register_builtin_partitioners
from fedbench.runtime.registry import FactoryRegistry


def build_algorithm_registry() -> FactoryRegistry[Algorithm]:
    registry: FactoryRegistry[Algorithm] = FactoryRegistry(
        group=f"{__package__}.algorithms",
        product_cls=Algorithm,  # type: ignore[type-abstract]
    )
    register_builtin_algorithms(registry)
    return registry


def build_partitioner_registry() -> FactoryRegistry[Partitioner]:
    registry: FactoryRegistry[Partitioner] = FactoryRegistry(
        group=f"{__package__}.partitioners",
        product_cls=Partitioner,  # type: ignore[type-abstract]
    )
    register_builtin_partitioners(registry)
    return registry


def build_evaluator_registry() -> FactoryRegistry[Evaluator]:
    registry = FactoryRegistry(
        group=f"{__package__}.evaluators",
        product_cls=Evaluator,  # type: ignore[type-abstract]
    )
    register_builtin_evaluators(registry)

    existing_keys: dict[str, str] = {}
    for entry in registry.metadata():
        evaluator = registry.call(entry.name)
        if evaluator.metadata.name != entry.name:
            raise ValueError(
                f"Evaluator's name in registry ({entry.name}) "
                f"does not match its declared name ({evaluator.metadata.name})."
            )
        keys = list(evaluator.get_metric_keys())
        if not keys:
            raise ValueError(
                f"Evaluator {evaluator.metadata.name} "
                f"does not declare any emitted metric keys."
            )
        for key in keys:
            if key in existing_keys:
                raise ValueError(
                    f"Metric key {key} is emitted multiple times, "
                    f"by both {evaluator.metadata.name} and {existing_keys[key]}."
                )
            existing_keys[key] = evaluator.metadata.name

    return registry
