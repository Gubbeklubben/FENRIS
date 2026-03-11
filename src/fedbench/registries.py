from types import MappingProxyType
from typing import Mapping

from fedbench.algorithms import register_builtin_algorithms
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import Partitioner
from fedbench.core.eval import Category, Evaluator
from fedbench.core.factory_registry import FactoryRegistry
from fedbench.evaluators import register_builtin_evaluators
from fedbench.partitioners import register_builtin_partitioners


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


def build_evaluator_registries() -> Mapping[str, FactoryRegistry[Evaluator]]:
    registries = {
        category.value: FactoryRegistry(
            group=f"{__package__}.evaluators.{category}",
            product_cls=Evaluator,  # type: ignore[type-abstract]
        )
        for category in Category
    }
    register_builtin_evaluators(registries)
    return MappingProxyType(registries)