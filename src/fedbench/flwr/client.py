from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict, ConfigRecord, MetricRecord,
)
from pandas.core.interchange.dataframe_protocol import DataFrame

from fedbench.flwr.serde import make_serde
from fedbench.algorithms import (
    Algorithm,
    Synthesizer,
    registry as algorithm_reg
)
from fedbench.config import Config
from fedbench.data import PartitionedDataset, load_csv, TableSchema
from fedbench.data.partitioners import registry as partitioner_reg
from fedbench.eval.context import EvalContext
from fedbench.eval.suite import EvaluationSuite

app = ClientApp()

config: Config | None = None
algorithm: type[Algorithm] | None = None
dataset: PartitionedDataset | None = None


def load_train_partition(context: Context) -> DataFrame:
    if dataset is None:
        raise RuntimeError(
            "Can not load train partition when dataset is not loaded."
        )
    # noinspection PyUnnecessaryCast
    partition_id: int = cast(int, context.node_config["partition-id"])
    return dataset.load_train_partition(partition_id)


def load_test_partition(context: Context) -> DataFrame:
    if dataset is None:
        raise RuntimeError(
            "Can not load test partition when dataset is not loaded."
        )
    # noinspection PyUnnecessaryCast
    partition_id: int = cast(int, context.node_config["partition-id"])
    return dataset.load_test_partition(partition_id)


def create_synthesizer() -> Synthesizer:
    if algorithm is None:
        raise RuntimeError(
            "Can not create synthesizer when algorithm is not loaded."
        )
    return algorithm.create_synthesizer()


@app.query("configure")
def configure(message: Message, context: Context) -> Message:
    global config
    cfg_record: ConfigRecord = message.content.config_records["config"]
    # noinspection PyUnnecessaryCast
    config = Config.parse_jsons(cast(str, cfg_record["jsons"]))

    global dataset
    df, schema = load_csv(config.data.dataset)
    partitioner_factory = partitioner_reg.load(config.data.partitioner)
    partitioner = partitioner_factory(**config.data.partitioner_kwargs)
    dataset = PartitionedDataset(
        df=df,
        schema=schema,
        partitioner=partitioner,
        test_size=config.test_size,
        seed=config.seed
    )
    global algorithm
    algorithm = algorithm_reg.load(config.algorithm)

    return Message(content=RecordDict(), reply_to=message)


@app.query("init")
def init(message: Message, context: Context) -> Message:
    train_df = load_train_partition(context)
    synthesizer = create_synthesizer()
    # noinspection PyUnnecessaryCast
    serializer, deserializer = make_serde(
        cast(Config, config).allow_pickle,
        synthesizer.arrays_to_ml_framework_map
    )
    request = deserializer(message)
    reply = synthesizer.init(request, train_df)
    return serializer(update=reply, reply_to=message)


@app.train()
def train(message: Message, context: Context) -> Message:
    train_df = load_train_partition(context)
    synthesizer = create_synthesizer()
    # noinspection PyUnnecessaryCast
    serializer, deserializer = make_serde(
        cast(Config, config).allow_pickle,
        synthesizer.arrays_to_ml_framework_map
    )
    request = deserializer(message)
    reply = synthesizer.train(request, train_df)
    return serializer(update=reply, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    train_df = load_train_partition(context)
    test_df = load_test_partition(context)
    synthesizer = create_synthesizer()
    # noinspection PyUnnecessaryCast
    serializer, deserializer = make_serde(
        cast(Config, config).allow_pickle,
        synthesizer.arrays_to_ml_framework_map
    )
    request = deserializer(message)
    synthetic_df = synthesizer.sample(request, config.num_synthetic_rows, config.seed)

    # noinspection PyUnnecessaryCast
    eval_ctx = EvalContext(
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        seed=config.seed,
        target_column=config.data.target_col,
        sensitive_columns=config.data.sensitive_cols,
        schema=cast(TableSchema, dataset.schema),
    )

    eval_suite = EvaluationSuite.default()
    #eval_suite = EvaluationSuite.with_evaluator_categories(config.metrics.run_categories)
    metrics = MetricRecord(eval_suite.evaluate(eval_ctx))

    return Message(content=RecordDict({"metrics": metrics}), reply_to=message)
