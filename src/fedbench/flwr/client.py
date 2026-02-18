from dataclasses import dataclass
from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict,
    ConfigRecord,
    MetricRecord,
)

from fedbench.algorithms import Algorithm, registry as algorithm_reg
from fedbench.config import Config
from fedbench.data import PartitionedDataset, load_csv
from fedbench.data.partitioners import registry as partitioner_reg
from fedbench.eval.context import EvalContext
from fedbench.eval.suite import EvaluationSuite
from fedbench.flwr.serde import make_serde, FlwrSerializer, FlwrDeserializer


app = ClientApp()


@dataclass(frozen=True)
class ClientContext:
    config: Config
    algorithm: type[Algorithm]
    dataset: PartitionedDataset
    to_flwr: FlwrSerializer
    from_flwr: FlwrDeserializer

client_ctx: ClientContext | None = None


def require_client_ctx() -> tuple[Config, type[Algorithm], PartitionedDataset,
        FlwrSerializer, FlwrDeserializer]:

    if client_ctx is None:
        raise RuntimeError("Client not configured.")

    return (client_ctx.config, client_ctx.algorithm, client_ctx.dataset,
            client_ctx.to_flwr, client_ctx.from_flwr)


def get_partition_id(flwr_context: Context) -> int:
    # noinspection PyUnnecessaryCast
    return cast(int, flwr_context.node_config["partition-id"])


@app.query("configure")
def configure(flwr_message: Message, flwr_context: Context) -> Message:
    cfg_record: ConfigRecord = flwr_message.content.config_records["config"]
    # noinspection PyUnnecessaryCast
    config = Config.parse_jsons(cast(str, cfg_record["jsons"]))

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
    algorithm = algorithm_reg.load(config.algorithm)
    to_flwr, from_flwr = make_serde(config.allow_pickle)

    global client_ctx
    client_ctx = ClientContext(
        config=config,
        algorithm=algorithm,
        dataset=dataset,
        to_flwr=to_flwr,
        from_flwr=from_flwr
    )
    return Message(content=RecordDict(), reply_to=flwr_message)


@app.query("init")
def init(flwr_message: Message, flwr_context: Context) -> Message:
    partition_id = get_partition_id(flwr_context)
    config, algorithm, dataset, to_flwr, from_flwr = require_client_ctx()
    train_df = dataset.load_train_partition(partition_id)

    synthesizer = algorithm.create_synthesizer()
    request = from_flwr(flwr_message, synthesizer.arrays_to_ml_framework_map)
    reply = synthesizer.init(request, train_df)

    return to_flwr(update=reply, reply_to=flwr_message)


@app.train()
def train(flwr_message: Message, flwr_context: Context) -> Message:
    partition_id = get_partition_id(flwr_context)
    config, algorithm, dataset, to_flwr, from_flwr = require_client_ctx()
    train_df = dataset.load_train_partition(partition_id)

    synthesizer = algorithm.create_synthesizer()
    request = from_flwr(flwr_message, synthesizer.arrays_to_ml_framework_map)
    reply = synthesizer.train(request, train_df)

    return to_flwr(update=reply, reply_to=flwr_message)


@app.evaluate()
def evaluate(flwr_message: Message, flwr_context: Context) -> Message:
    partition_id = get_partition_id(flwr_context)
    config, algorithm, dataset, to_flwr, from_flwr = require_client_ctx()
    train_df = dataset.load_train_partition(partition_id)
    test_df = dataset.load_test_partition(partition_id)

    synthesizer = algorithm.create_synthesizer()
    request = from_flwr(flwr_message, synthesizer.arrays_to_ml_framework_map)
    synthetic_df = synthesizer.sample(
        request,
        config.num_synthetic_rows or len(train_df),
        config.seed
    )
    # noinspection PyUnnecessaryCast
    eval_ctx = EvalContext(
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        seed=config.seed,
        target_column=config.data.target_col,
        sensitive_columns=config.data.sensitive_cols,
        schema=dataset.schema,
    )
    eval_suite = EvaluationSuite.default()
    #eval_suite = EvaluationSuite.with_evaluator_categories(config.metrics.run_categories)
    metrics = MetricRecord()
    for key, value in eval_suite.evaluate(eval_ctx).items():
        metrics[key] = value

    return Message(
        content=RecordDict({"metrics": metrics}),
        reply_to=flwr_message
    )
