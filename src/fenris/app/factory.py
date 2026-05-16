import functools
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from pandas import DataFrame

from fenris.app.plugins import Registry, plugins
from fenris.core.algorithm import Coordinator, Synthesizer
from fenris.core.data import Partitioner, load_csv
from fenris.core.eval import Category, EvaluationSuite, Evaluator


# Wrap up loading in a partial for easy replay in client subprocs.
# I imagine loader at some point should become pluggable, to enable other
# sources than local csv.
def create_df_loader(dataset: str) -> Callable[[], DataFrame]:
    return functools.partial(load_csv, dataset)


def create_synthesizer(
    name: str, kwargs: dict[str, Any], registry: Registry[Synthesizer] | None = None
) -> Synthesizer:

    registry = registry or plugins.synthesizers.registry
    cls = registry.load(name)
    # noinspection PyArgumentList
    return cls(**kwargs)


def create_coordinator(
    name: str, kwargs: dict[str, Any], registry: Registry[Coordinator] | None = None
) -> Coordinator:

    registry = registry or plugins.coordinators.registry
    cls = registry.load(name)
    # noinspection PyArgumentList
    return cls(**kwargs)


def create_partitioner(
    name: str, kwargs: dict[str, Any], registry: Registry[Partitioner] | None = None
) -> Partitioner:

    registry = registry or plugins.partitioners.registry
    cls = registry.load(name)
    # noinspection PyArgumentList
    return cls(**kwargs)


def create_evaluator(
    name: str, registry: Registry[Evaluator] | None = None
) -> Evaluator:
    registry = registry or plugins.evaluators.registry
    cls = registry.load(name)
    return cls()


def create_evaluators(
    categories: Iterable[Category], registry: Registry[Evaluator] | None = None
) -> Iterable[Evaluator]:

    registry = registry or plugins.evaluators.registry
    instances = defaultdict(list)
    # noinspection PyTypeChecker
    for name in registry:
        evaluator = create_evaluator(name, registry)
        instances[evaluator.EVALUATOR_SPEC.category].append(evaluator)
    for category in categories:
        for evaluator in instances[category]:
            yield evaluator


def create_evaluation_suite(
    categories: Iterable[Category], registry: Registry[Evaluator] | None = None
) -> EvaluationSuite:

    return EvaluationSuite(create_evaluators(categories, registry))
