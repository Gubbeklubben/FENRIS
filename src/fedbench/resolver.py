import functools
from collections.abc import Mapping

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import Partitioner, load_csv
from fedbench.core.eval import Evaluator, EvaluationSuite
from fedbench.core.factory_registry import FactoryRegistry
from fedbench.core.runcontext import Components


def resolve_components(
        config: Config,
        algorithms: FactoryRegistry[Algorithm],
        partitioners: FactoryRegistry[Partitioner],
        evaluators: Mapping[str, FactoryRegistry[Evaluator]],) -> Components:

    df_loader = functools.partial(load_csv, config.data.dataset)

    algorithm = algorithms.call(
        config.algorithm,
        config.algorithm_kwargs,
    )
    partitioner = partitioners.call(
        config.data.partitioner,
        config.data.partitioner_kwargs,
    )

    if not config.metrics.run_categories:
        eval_suite = EvaluationSuite.default(evaluators)
    else:
        eval_suite = EvaluationSuite.with_evaluator_categories(
            evaluators, config.metrics.run_categories,
        )
    return Components(df_loader, algorithm, partitioner, eval_suite)


