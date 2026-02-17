from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict, ConfigRecord,
)
from pandas.core.interchange.dataframe_protocol import DataFrame

from fedbench._flwr.serde import from_flwr, to_flwr
from fedbench.algorithms import (
    Algorithm,
    Synthesizer,
    registry as algorithm_reg
)
from fedbench.config import Config
from fedbench.data import PartitionedDataset, load_csv
from fedbench.data.partitioners import registry as partitioner_reg

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
    data = load_train_partition(context)
    request = from_flwr(message)
    synthesizer = create_synthesizer()
    reply = synthesizer.init(request, data)

    return to_flwr(
        update=reply,
        reply_to=message,
        non_array_protocol=algorithm.requires_non_array_protocol())


@app.train()
def train(message: Message, context: Context) -> Message:
    data = load_train_partition(context)
    request = from_flwr(message)
    synthesizer = create_synthesizer()
    reply = synthesizer.train(request, data)

    return to_flwr(
        update=reply,
        reply_to=message,
        non_array_protocol=algorithm.requires_non_array_protocol())


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    return Message(content=RecordDict(), reply_to=message)
