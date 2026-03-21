from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from flwr.common import RecordDict

from fedbench.config import Config
from fedbench.core.algorithm import Synthesizer
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema
from fedbench.core.eval import EvaluationSuite
from fedbench.flwr.namespace import Namespace
from fedbench.flwr.rdict import RDictNamespaceView
from fedbench.flwr.serde import FlwrSerde, Pickle
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

_config: Config | None = None
_dataset: PartitionedDataset | None = None


@dataclass(frozen=True)
class ClientContext:
    config: Config
    dataset: PartitionedDataset
    synthesizer_factory: Callable[[], Synthesizer]
    eval_suite: EvaluationSuite
    serde: FlwrSerde
    framework_cache: RDictNamespaceView
    artifacts_cache: RDictNamespaceView
    synthesizer_cache: RDictNamespaceView


def build_client_context(flwr_cache: RecordDict) -> ClientContext:
    framework_cache = Namespace.FRAMEWORK.view(flwr_cache)
    config = _get_config(framework_cache)
    dataset = _get_dataset(config)

    algorithm = create_algorithm(config, build_algorithm_registry())
    eval_suite = create_evaluation_suite(config, build_evaluator_registries())
    serde = FlwrSerde(
        object_serde=Pickle(disabled=config.disable_pickle),
        default_arrays_map=algorithm.synthesizer_spec.arrays_to_ml_framework_map,
    )
    return ClientContext(
        config,
        dataset,
        algorithm.synthesizer_spec.factory,
        eval_suite,
        serde,
        framework_cache,
        Namespace.GLOBAL_INIT_ARTIFACTS.view(flwr_cache),
        Namespace.SYNTHESIZER.view(flwr_cache),
    )


def _get_config(cache: RDictNamespaceView) -> Config:
    global _config
    if _config is not None:
        return _config

    config = cache.config_records.get("config", None)
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
        seed=config.seed.partitioning,
    )
    return _dataset
