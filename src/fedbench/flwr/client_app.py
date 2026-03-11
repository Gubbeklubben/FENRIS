from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    ConfigRecord,
    Context,
    Message,
    RecordDict,
)

from fedbench.component_factory import (
    create_df_loader,
    create_algorithm,
    create_evaluation_suite,
    create_partitioner,
)
from fedbench.config import Config
from fedbench.core.data import PartitionedDataset
from fedbench.core.data.schemas import infer_schema
from fedbench.flwr.client import FlwrClient
from fedbench.flwr.serde import (
    from_flwr_pickle,
    to_flwr_no_pickle,
    to_flwr_pickle,
)
from fedbench.registries import (
    build_algorithm_registry,
    build_evaluator_registries,
    build_partitioner_registry,
)

config: Config | None = None
flwr_client: FlwrClient | None = None
app = ClientApp()


def require_context() -> tuple[Config, FlwrClient]:
    if config is None or flwr_client is None:
        raise RuntimeError("Client not properly configured.")
    return config, flwr_client


@app.query("config")
def recv_config(flwr_message: Message, _: Context) -> Message:
    cfg_record: ConfigRecord = flwr_message.content.config_records["config"]
    global config
    # noinspection PyUnnecessaryCast
    config = Config.parse_jsons(cast(str, cfg_record["jsons"]))
    return Message(content=RecordDict(), reply_to=flwr_message)


@app.query("artifacts")
def recv_artifacts(flwr_message: Message, _: Context) -> Message:
    if config is None:
        raise RuntimeError("Missing config.")

    df_loader = create_df_loader(config)
    algorithm = create_algorithm(config, build_algorithm_registry())
    partitioner = create_partitioner(config, build_partitioner_registry())
    eval_suite = create_evaluation_suite(config, build_evaluator_registries())

    synthesizer_spec = algorithm.synthesizer_spec
    if flwr_message.has_content():
        artifacts = from_flwr_pickle(
            flwr_message,
            synthesizer_spec.arrays_to_ml_framework_map,
        )
    else:
        artifacts = None

    df = df_loader()
    schema = infer_schema(df)

    dataset = PartitionedDataset(
        df=df,
        schema=schema,
        partitioner=partitioner,
        test_size=config.test_size,
        seed=config.seed,
    )
    global flwr_client
    flwr_client = FlwrClient(
        dataset,
        synthesizer_spec,
        artifacts,
        eval_suite,
        to_flwr_no_pickle if config.disable_pickle else to_flwr_pickle,
        from_flwr_pickle,
    )
    return Message(content=RecordDict(), reply_to=flwr_message)


@app.query("fed_init")
def fed_init(flwr_message: Message, flwr_context: Context) -> Message:
    cfg, client = require_context()
    return client.fed_init(cfg.seed, flwr_message, flwr_context)


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
