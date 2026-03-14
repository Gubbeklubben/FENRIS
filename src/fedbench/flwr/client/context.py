from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

from flwr.common import RecordDict

from fedbench.config import Config
from fedbench.core.algorithm import ComponentSpec, Synthesizer
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema
from fedbench.core.eval import EvaluationSuite
from fedbench.core.update import Update
from fedbench.flwr.client.cache_manager import CacheManager, Namespace
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


@dataclass(frozen=True)
class ClientContext:
    config: Config
    dataset: PartitionedDataset
    synthesizer_spec: ComponentSpec[Synthesizer]
    synthesizer_artifacts: Update | None
    eval_suite: EvaluationSuite
    serde: FlwrSerde

    @contextmanager
    def use_synthesizer_cache(
        self, cache_mgr: CacheManager
    ) -> Generator[Update, None, None]:

        cache = self.serde.from_flwr(
            cache_mgr.get_cache(Namespace.SYNTHESIZER),
            self.synthesizer_spec.arrays_to_ml_framework_map,
        )
        try:
            yield cache
        finally:
            cache_mgr.set_cache(Namespace.SYNTHESIZER, self.serde.to_flwr(cache))


_config: Config | None = None
_dataset: PartitionedDataset | None = None


def build_client_context(cache_mgr: CacheManager) -> ClientContext:
    config = _get_config(cache_mgr.get_cache(Namespace.FRAMEWORK))
    dataset = _get_dataset(config)

    algorithm = create_algorithm(config, build_algorithm_registry())
    eval_suite = create_evaluation_suite(config, build_evaluator_registries())
    serde = FlwrSerde(object_serde=Pickle(config.disable_pickle))

    artifacts_rdict = cache_mgr.get_cache(Namespace.GLOBAL_INIT_ARTIFACTS)

    if artifacts_rdict:
        artifacts = serde.from_flwr(
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
        serde=serde,
    )


def _get_config(cache: RecordDict) -> Config:
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
        seed=config.seed,
    )
    return _dataset
