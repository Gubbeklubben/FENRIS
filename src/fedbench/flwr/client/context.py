from dataclasses import dataclass
from pathlib import Path
from typing import cast

from flwr.app import RecordDict

import fedbench.app.factory as factory
from fedbench.app.run.partitioned_dataset import PartitionedDataset
from fedbench.config import Config
from fedbench.core.algorithm import Synthesizer
from fedbench.core.data.schemas import load_or_infer_schema
from fedbench.core.eval import EvaluationSuite
from fedbench.flwr.namespace import Namespace
from fedbench.flwr.rdict import RDictNamespaceView
from fedbench.flwr.serde import FlwrSerde, Pickle

_config: Config | None = None
_dataset: PartitionedDataset | None = None


@dataclass(frozen=True)
class ClientContext:
    config: Config
    dataset: PartitionedDataset
    synthesizer: Synthesizer
    eval_suite: EvaluationSuite
    serde: FlwrSerde
    framework_storage: RDictNamespaceView
    artifacts_storage: RDictNamespaceView
    synthesizer_storage: RDictNamespaceView


def build_client_context(flwr_storage: RecordDict) -> ClientContext:
    framework_storage = Namespace.FRAMEWORK.view(flwr_storage)
    config = _get_config(framework_storage)
    dataset = _get_dataset(config)

    synthesizer = factory.create_synthesizer(
        config.synthesizer, config.synthesizer_kwargs
    )
    eval_suite = factory.create_evaluation_suite(set(config.metrics.run_categories))
    serde = FlwrSerde(
        object_serde=Pickle(disabled=config.disable_pickle),
        default_arrays_target=synthesizer.arrays_target,
    )
    return ClientContext(
        config,
        dataset,
        synthesizer,
        eval_suite,
        serde,
        framework_storage,
        Namespace.GLOBAL_INIT_ARTIFACTS.view(flwr_storage),
        Namespace.SYNTHESIZER.view(flwr_storage),
    )


def _get_config(storage: RDictNamespaceView) -> Config:
    global _config
    if _config is not None:
        return _config

    config = storage.config_records.get("config", None)
    if config is None:
        raise RuntimeError("Missing config, can not build client context.")

    # noinspection PyUnnecessaryCast
    _config = Config.parse_jsons(cast(str, config["jsons"]))
    return _config


def _get_dataset(config: Config) -> PartitionedDataset:
    global _dataset
    if _dataset is not None:
        return _dataset

    df_loader = factory.create_df_loader(config.data.dataset)
    df = df_loader()
    schema = load_or_infer_schema(Path(config.data.schema), df)

    partitioner = factory.create_partitioner(
        config.data.partitioner,
        config.data.partitioner_kwargs,
    )
    _dataset = PartitionedDataset(
        df=df,
        schema=schema,
        partitioner=partitioner,
        test_size=config.test_size,
        seed=config.seed.partitioning,
    )
    return _dataset
