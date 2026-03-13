from dataclasses import dataclass
from typing import cast

from fedbench.config import Config
from fedbench.core.algorithm import ComponentSpec, Synthesizer
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema
from fedbench.core.eval import EvaluationSuite
from fedbench.core.update import Update
from fedbench.flwr.client_cache_wrapper import ClientCacheWrapper
from fedbench.flwr.serde import (
    FlwrDeserializer,
    FlwrSerializer,
    from_flwr_pickle,
    to_flwr_no_pickle,
    to_flwr_pickle,
)
from fedbench.runtime.component_factory import (
    create_algorithm,
    create_df_loader,
    create_evaluation_suite,
    create_partitioner,
)
from fedbench.runtime.registry_builder import (
    build_algorithm_registry,
    build_evaluator_registries,
    build_partitioner_registry,
)


@dataclass(frozen=True)
class ClientContext:
    config: Config
    dataset: PartitionedDataset
    synthesizer_spec: ComponentSpec[Synthesizer]
    synthesizer_artifacts: Update | None
    eval_suite: EvaluationSuite
    to_flwr: FlwrSerializer
    from_flwr: FlwrDeserializer


_config: Config | None = None
_dataset: PartitionedDataset | None = None


def build_client_context(cache: ClientCacheWrapper) -> ClientContext:
    config = _get_config(cache)
    dataset = _get_dataset(config)

    algorithm = create_algorithm(config, build_algorithm_registry())
    eval_suite = create_evaluation_suite(config, build_evaluator_registries())
    to_flwr = to_flwr_no_pickle if config.disable_pickle else to_flwr_pickle
    from_flwr = from_flwr_pickle

    artifacts_rdict = cache.get_artifacts()

    if artifacts_rdict is not None:
        artifacts = from_flwr(
            artifacts_rdict, algorithm.synthesizer_spec.arrays_to_ml_framework_map
        )
    else:
        artifacts = None

    return ClientContext(
        config=config,
        dataset=dataset,
        synthesizer_spec=algorithm.synthesizer_spec,
        synthesizer_artifacts=artifacts,
        eval_suite=eval_suite,
        to_flwr=to_flwr,
        from_flwr=from_flwr,
    )


def _get_config(cache: ClientCacheWrapper) -> Config:
    global _config
    if _config is not None:
        return _config

    config = cache.get_config()
    if config is None:
        raise RuntimeError("Missing config, can not build client context.")

    # noinspection PyUnnecessaryCast
    _config = Config.parse_jsons(cast(str, config["jsons"]))
    return _config


def _get_dataset(config: Config) -> PartitionedDataset:
    global _dataset
    if _dataset is not None:
        return _dataset

    df_loader = create_df_loader(config)
    df = df_loader()
    schema = infer_schema(df)
    partitioner = create_partitioner(config, build_partitioner_registry())

    _dataset = PartitionedDataset(
        df=df,
        schema=schema,
        partitioner=partitioner,
        test_size=config.test_size,
        seed=config.seed,
    )
    return _dataset
