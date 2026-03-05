from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict,
    ConfigRecord,
    MetricRecord,
)

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema
from fedbench.core.eval import EvalContext, EvaluationSuite
from fedbench.core.logger import log_warning
from fedbench.flwr.serde import (
    FlwrSerializer, FlwrDeserializer,
    to_flwr_pickle, from_flwr_pickle, to_flwr_disable_pickle
)
from fedbench.registries import (
    build_algorithm_registry,
    build_partitioner_registry,
    build_evaluator_registries,
)
from fedbench.resolver import resolve_components


class FedbenchClient:
    def __init__(
            self,
            dataset: PartitionedDataset,
            algorithm: Algorithm,
            eval_suite: EvaluationSuite,
            to_flwr: FlwrSerializer,
            from_flwr: FlwrDeserializer,) -> None:

        self._dataset = dataset
        self._algorithm = algorithm
        self._eval_suite = eval_suite
        self._to_flwr = to_flwr
        self._from_flwr = from_flwr

    def init(
            self,
            seed: int,
            flwr_message: Message,
            flwr_context: Context,) -> Message:

        partition_id = self._get_partition_id(flwr_context)
        train_df = self._dataset.load_train_partition(partition_id)
        synthesizer = self._algorithm.create_synthesizer()

        request = self._from_flwr(
            flwr_message,
            synthesizer.arrays_to_ml_framework_map,
        )
        reply = synthesizer.fed_init(request, seed, self._dataset.schema, train_df)
        return self._to_flwr(update=reply, reply_to=flwr_message)


    def train(self, flwr_message: Message, flwr_context: Context) -> Message:
        partition_id = self._get_partition_id(flwr_context)
        train_df = self._dataset.load_train_partition(partition_id)
        synthesizer = self._algorithm.create_synthesizer()

        request = self._from_flwr(
            flwr_message,
            synthesizer.arrays_to_ml_framework_map,
        )
        reply = synthesizer.train(request, train_df)
        return self._to_flwr(update=reply, reply_to=flwr_message)

    def evaluate(
            self,
            flwr_message: Message,
            flwr_context: Context,
            num_synthetic_rows: int | None,
            seed: int,
            target_column: str | None,
            sensitive_columns: tuple[str, ...] | None,) -> Message:

        partition_id = self._get_partition_id(flwr_context)
        train_df = self._dataset.load_train_partition(partition_id)
        test_df = self._dataset.load_test_partition(partition_id)

        synthesizer = self._algorithm.create_synthesizer()
        request = self._from_flwr(
            flwr_message,
            synthesizer.arrays_to_ml_framework_map,
        )
        synthetic_df = synthesizer.sample(
            request,
            num_synthetic_rows or len(train_df),
            seed,
        )
        if synthetic_df.empty:
            log_warning(__name__, f"Recv empty sample from {synthesizer}.")
            return Message(
                content=RecordDict({"metrics": MetricRecord()}),
                reply_to=flwr_message,
            )
        eval_ctx = EvalContext(
            train_df=train_df,
            test_df=test_df,
            synthetic_df=synthetic_df,
            seed=seed,
            target_column=target_column,
            sensitive_columns=sensitive_columns,
            schema=self._dataset.schema,
        )
        metrics = MetricRecord()
        for key, value in self._eval_suite.evaluate(eval_ctx).items():
            metrics[key] = value

        return Message(
            content=RecordDict({"metrics": metrics}),
            reply_to=flwr_message,
        )

    @staticmethod
    def _get_partition_id(flwr_context: Context) -> int:
        # noinspection PyUnnecessaryCast
        return cast(int, flwr_context.node_config["partition-id"])


config: Config | None = None
fedbench_client: FedbenchClient | None = None
app = ClientApp()


def require_context() -> tuple[Config, FedbenchClient]:
    if config is None or fedbench_client is None:
        raise RuntimeError("Client not properly configured.")
    return config, fedbench_client


@app.query("configure")
def configure(flwr_message: Message, _: Context) -> Message:
    cfg_record: ConfigRecord = flwr_message.content.config_records["config"]
    global config
    # noinspection PyUnnecessaryCast
    config = Config.parse_jsons(cast(str, cfg_record["jsons"]))

    components = resolve_components(
        config,
        build_algorithm_registry(),
        build_partitioner_registry(),
        build_evaluator_registries(),
    )
    df = components.df_loader()
    schema = infer_schema(df)

    dataset = PartitionedDataset(
        df=df,
        schema=schema,
        partitioner=components.partitioner,
        test_size=config.test_size,
        seed=config.seed,
    )
    global fedbench_client
    fedbench_client = FedbenchClient(
        dataset,
        components.algorithm,
        components.eval_suite,
        to_flwr_disable_pickle if config.disable_pickle else to_flwr_pickle,
        from_flwr_pickle,
    )
    return Message(content=RecordDict(), reply_to=flwr_message)


@app.query("init")
def init(flwr_message: Message, flwr_context: Context) -> Message:
    cfg, client = require_context()
    return client.init(cfg.seed, flwr_message, flwr_context)


@app.train()
def train(flwr_message: Message, flwr_context: Context) -> Message:
    _, client = require_context()
    return client.train(flwr_message, flwr_context)


@app.evaluate()
def evaluate(flwr_message: Message, flwr_context: Context) -> Message:
    cfg, client = require_context()
    return client.evaluate(
        flwr_message,
        flwr_context,
        cfg.num_synthetic_rows,
        cfg.seed,
        cfg.data.target_col,
        cfg.data.sensitive_cols,
    )
