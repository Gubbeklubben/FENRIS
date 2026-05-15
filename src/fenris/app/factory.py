import functools
import inspect
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from pandas import DataFrame

from fenris.app.registry import Group, Registry
from fenris.core.algorithm import Coordinator, Synthesizer
from fenris.core.component import Component
from fenris.core.data import Partitioner, load_csv
from fenris.core.eval import Category, EvaluationSuite, Evaluator


# Wrap up loading in a partial for easy replay in client subprocs.
# I imagine loader at some point should become pluggable, to enable other
# sources than local csv.
def create_df_loader(dataset: str) -> Callable[[], DataFrame]:
    return functools.partial(load_csv, dataset)


def create_synthesizer(
    name: str, kwargs: dict[str, Any], registry: Registry | None = None
) -> Synthesizer:

    registry = registry or Group.SYNTHESIZERS.get_registry()
    return _create_component(registry, name, kwargs, Synthesizer)  # type: ignore[type-abstract]


def create_coordinator(
    name: str, kwargs: dict[str, Any], registry: Registry | None = None
) -> Coordinator:

    registry = registry or Group.COORDINATORS.get_registry()
    return _create_component(registry, name, kwargs, Coordinator)  # type: ignore[type-abstract]


def create_partitioner(
    name: str, kwargs: dict[str, Any], registry: Registry | None = None
) -> Partitioner:

    registry = registry or Group.PARTITIONERS.get_registry()
    return _create_component(registry, name, kwargs, Partitioner)  # type: ignore[type-abstract]


def create_evaluator(name: str, registry: Registry | None = None) -> Evaluator:
    registry = registry or Group.EVALUATORS.get_registry()
    return _create_component(registry, name, {}, Evaluator)  # type: ignore[type-abstract]


def create_evaluators(
    categories: Iterable[Category], registry: Registry | None = None
) -> Iterable[Evaluator]:

    registry = registry or Group.EVALUATORS.get_registry()
    evaluators = defaultdict(list)
    # noinspection PyTypeChecker
    for name in registry:
        evaluator = create_evaluator(name, registry)
        evaluators[evaluator.EVALUATOR_SPEC.category].append(evaluator)
    for category in categories:
        for evaluator in evaluators[category]:
            yield evaluator


def create_evaluation_suite(
    categories: Iterable[Category], registry: Registry | None = None
) -> EvaluationSuite:

    return EvaluationSuite(create_evaluators(categories, registry))


def _create_component[T: Component](
    registry: Registry,
    name: str,
    kwargs: dict[str, Any],
    product_cls: type[T],
) -> T:

    factory = registry.load(name)

    if inspect.isclass(factory) and inspect.isabstract(factory):
        raise TypeError(f"{factory} is an abstract class.")

    if not callable(factory):
        raise TypeError(f"{factory} is not callable.")

    instance = factory(**kwargs)
    if not isinstance(instance, product_cls):
        raise TypeError(f"Unexpected type {type(instance)} produced by factory {name}")

    if instance.name != name:
        raise ValueError(
            f"Component name does not match registry key, "
            f"type: {product_cls}, name: {instance.name}, key: {name}."
        )
    return instance
