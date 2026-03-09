import functools
from collections.abc import Mapping, Callable

from pandas import DataFrame

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import Partitioner, load_csv
from fedbench.core.eval import EvaluationSuite, Evaluator
from fedbench.core.factory_registry import FactoryRegistry


# Wrap up loading in a partial for easy replay in client subprocs.
def resolve_df_loader(config: Config) -> Callable[[], DataFrame]:
    return functools.partial(load_csv, config.data.dataset)


def resolve_algorithm(
    config: Config,
    registry: FactoryRegistry[Algorithm],
) -> Algorithm:

    return registry.call(
        config.algorithm,
        config.algorithm_kwargs,
    )


def resolve_partitioner(
    config: Config,
    registry: FactoryRegistry[Partitioner],
) -> Partitioner:

    return registry.call(
        config.data.partitioner,
        config.data.partitioner_kwargs,
    )


def resolve_evaluators(
    config: Config,
    registries: Mapping[str, FactoryRegistry[Evaluator]],
) -> EvaluationSuite:

    if not config.metrics.run_categories:
        return EvaluationSuite.default(registries)

    return EvaluationSuite.with_evaluator_categories(
        registries,
        config.metrics.run_categories,
    )