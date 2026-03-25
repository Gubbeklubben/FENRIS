from fedbench.core.algorithm import Algorithm, Coordinator
from fedbench.core.data import Partitioner
from fedbench.core.eval import Evaluator
from fedbench.runtime.registry import FactoryRegistry

_ROOT_PKG = __package__.split(".")[0]


def build_algorithm_registry() -> FactoryRegistry[Algorithm]:
    return FactoryRegistry(
        group=f"{_ROOT_PKG}.algorithms",
        product_cls=Algorithm,  # type: ignore[type-abstract]
    )


def build_coordinator_registry() -> FactoryRegistry[Coordinator]:
    return FactoryRegistry(
        group=f"{_ROOT_PKG}.coordinators",
        product_cls=Coordinator,  # type: ignore[type-abstract]
    )


def build_partitioner_registry() -> FactoryRegistry[Partitioner]:
    return FactoryRegistry(
        group=f"{_ROOT_PKG}.partitioners",
        product_cls=Partitioner,  # type: ignore[type-abstract]
    )


def build_evaluator_registry() -> FactoryRegistry[Evaluator]:
    registry = FactoryRegistry(
        group=f"{_ROOT_PKG}.evaluators",
        product_cls=Evaluator,  # type: ignore[type-abstract]
    )

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
