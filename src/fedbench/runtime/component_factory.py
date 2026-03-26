import functools
from collections.abc import Callable

from pandas import DataFrame

from fedbench.config import Config
from fedbench.core.algorithm import Coordinator, Synthesizer
from fedbench.core.data import PartitionedDataset, Partitioner, load_csv
from fedbench.core.eval import CentralizedEvalContext, EvaluationSuite, Evaluator
from fedbench.runtime.registry import FactoryRegistry


# Wrap up loading in a partial for easy replay in client subprocs.
def create_df_loader(config: Config) -> Callable[[], DataFrame]:
    return functools.partial(load_csv, config.data.dataset)


def create_synthesizer(
    config: Config,
    registry: FactoryRegistry[Synthesizer],
) -> Synthesizer:

    return registry.call(
        config.synthesizer,
        config.synthesizer_kwargs,
    )


def create_coordinator(
    config: Config,
    registry: FactoryRegistry[Coordinator],
) -> Coordinator:

    return registry.call(
        config.coordinator,
        config.coordinator_kwargs,
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


def create_centralized_eval_ctx(
    config: Config,
    dataset: PartitionedDataset,
    synthetic_df: DataFrame,
) -> CentralizedEvalContext:
    return CentralizedEvalContext(
        synthetic_df=synthetic_df,
        holdout_df=dataset.load_global_holdout(),
        client_train_df=dataset.load_all_train_data(),
        target_column=config.data.target_col,
        sensitive_columns=config.data.sensitive_cols,
        schema=dataset.schema,
        seed=config.seed.evaluation,
    )
