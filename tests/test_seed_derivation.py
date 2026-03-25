"""Verify that pipeline functions pass the correct derived seeds (§23.2)."""

import random
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fedbench.config import Config, DataConfig, SeedConfig
from fedbench.core.algorithm import GlobalInitArtifacts
from fedbench.core.data.schemas import infer_schema
from fedbench.runtime.pipeline import (
    global_evaluate,
    global_init,
    global_sample,
    load_dataset,
)

SEED = random.randint(1, (2**32) - 3)
SEEDS = SeedConfig.from_master(SEED)


@pytest.fixture
def config():
    return Config(
        algorithm="fed_hello",
        coordinator="MISSING",
        data=DataConfig(dataset="/dev/null", partitioner="iid_partitioner"),
        seed=SeedConfig.from_master(SEED),
        num_synthetic_rows=20,
    )


@pytest.fixture
def ctx(config):
    from fedbench.runtime.eventbus import EventBus
    from fedbench.runtime.runcontext import RunContext
    from fedbench.runtime.scalability_collector import ScalabilityCollector

    return RunContext("test", config, EventBus(), ScalabilityCollector())


def test_load_dataset_seed(ctx):
    df = pd.DataFrame({"a": range(20), "b": range(20)})
    ctx.df_loader = lambda: df

    partitioner = MagicMock()
    partitioner.num_partitions = 3
    ctx.partitioner = partitioner

    load_dataset(ctx)

    assert ctx.dataset._seed == SEEDS.partitioning


def test_global_init_seed(ctx):
    df = pd.DataFrame({"a": range(20)})
    schema = infer_schema(df)
    ctx.dataset = MagicMock(
        schema=schema, load_all_train_data=MagicMock(return_value=df)
    )

    algorithm = MagicMock()
    algorithm.global_init.return_value = None
    ctx.algorithm = algorithm

    global_init(ctx)

    algorithm.global_init.assert_called_once_with(SEEDS.init, schema, df)


def test_global_sample_seed(ctx):
    ctx.aggregated_state = MagicMock()
    ctx.global_init_artifacts = GlobalInitArtifacts(None, None)
    ctx.algorithm = MagicMock()

    synthesizer = MagicMock()
    with patch(
        "fedbench.runtime.pipeline.create_synthesizer", return_value=synthesizer
    ):
        global_sample(ctx)

    synthesizer.sample.assert_called_once()
    assert synthesizer.sample.call_args[0][2] == SEEDS.sampling


def test_global_evaluate_seed(ctx):
    df = pd.DataFrame({"a": [1, 2]})
    schema = infer_schema(df)

    ctx.dataset = MagicMock(
        schema=schema,
        load_global_holdout=MagicMock(return_value=df),
        load_all_train_data=MagicMock(return_value=df),
    )
    ctx.synthetic_df = df
    ctx.eval_suite = MagicMock()

    global_evaluate(ctx)

    eval_ctx = ctx.eval_suite.global_evaluate.call_args[0][0]
    assert eval_ctx.seed == SEEDS.evaluation
