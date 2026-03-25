import functools
from collections.abc import Callable

from pandas import DataFrame

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm, Coordinator, Synthesizer
from fedbench.core.data import Partitioner, load_csv
from fedbench.core.eval import EvaluationSuite, Evaluator
from fedbench.core.payload import Payload
from fedbench.runtime.registry import FactoryRegistry


# Wrap up loading in a partial for easy replay in client subprocs.
def create_df_loader(config: Config) -> Callable[[], DataFrame]:
    return functools.partial(load_csv, config.data.dataset)


def create_algorithm(
    config: Config,
    registry: FactoryRegistry[Algorithm],
) -> Algorithm:

    return registry.call(
        config.algorithm,
        config.algorithm_kwargs,
    )


def create_partitioner(
    config: Config,
    registry: FactoryRegistry[Partitioner],
) -> Partitioner:

    return registry.call(
        config.data.partitioner,
        config.data.partitioner_kwargs,
    )


def create_evaluation_suite(
    config: Config,
    registry: FactoryRegistry[Evaluator],
) -> EvaluationSuite:

    if not config.metrics.run_categories:
        return EvaluationSuite.default(registry)

    return EvaluationSuite.with_evaluator_categories(
        registry,
        config.metrics.run_categories,
    )


def create_coordinator(
    factory: Callable[[], Coordinator],
    artifacts: Payload | None,
) -> Coordinator:

    instance = factory()
    if not isinstance(instance, Coordinator):
        raise TypeError(f"{instance} is not a Coordinator.")

    if artifacts is not None:
        instance.attach_global_init_artifacts(artifacts)

    return instance


def create_synthesizer(
    factory: Callable[[], Synthesizer],
    artifacts: Payload | None,
    client_cache: Payload | None,
) -> Synthesizer:

    instance = factory()
    if not isinstance(instance, Synthesizer):
        raise TypeError(f"{instance} is not a Synthesizer.")

    if artifacts is not None:
        instance.attach_global_init_artifacts(artifacts)

    if client_cache is not None:
        instance.attach_client_cache(client_cache)

    return instance
