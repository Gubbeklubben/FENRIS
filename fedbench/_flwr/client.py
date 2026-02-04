from logging import DEBUG
from typing import cast
from numpy.typing import NDArray

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict,
    ArrayRecord,
    MetricRecord,
    Array,
    ConfigRecord,
)

from fedbench._plugins import load_synthesizer_factory
from fedbench.common import InitRequest, log, MLRuntime, TrainRequest
from fedbench.synthesizer import Synthesizer
from fedbench._flwr.common import from_array_record

app = ClientApp()
synthesizer_factory = None


def to_array_record(statistics: dict[str, NDArray]) -> ArrayRecord:
    return ArrayRecord({k: Array(ndarray) for k, ndarray in statistics.items()})


def get_synthesizer(config: ConfigRecord | None = None) -> Synthesizer:
    global synthesizer_factory
    if synthesizer_factory is None:
        if config is None:
            raise ValueError("Can not load synthesizer factory without config.")
        # noinspection PyUnnecessaryCast
        algorithm_name: str = cast(str, config["algorithm-name"])
        synthesizer_factory = load_synthesizer_factory(algorithm_name)
    return synthesizer_factory()


@app.query("init")
def init(message: Message, context: Context) -> Message:
    config = message.content.config_records["config"]
    synthesizer = get_synthesizer(config)
    response = synthesizer.init(InitRequest(message.metadata.dst_node_id, None))

    content = RecordDict()
    if response.statistics is not None:
        record = to_array_record(response.statistics)
        content["init"] = record

    return Message(
        content=content,
        reply_to=message
    )

@app.train()
def train(message: Message, context: Context) -> Message:
    # Load data
    # Call synthesizer factory
    # Call synthesizer.train
    # Get synthesizer weights / other relevant stuff
    # Convert to Flower Message and return it
    synthesizer = get_synthesizer()
    arrays = message.content.array_records["arrays"]

    request = TrainRequest(
        client_id=message.metadata.dst_node_id,
        model_state=from_array_record(arrays, synthesizer.ml_runtime),
        config=None
    )
    response = synthesizer.train(request)

    content = RecordDict()
    # flwr requires "num-examples" to be present
    metrics = MetricRecord({"num-examples": response.num_examples})

    if response.model_state is not None:
        content["arrays"] = ArrayRecord(response.model_state)

    if response.metrics is not None:
        for k, v in response.metrics.items():
            metrics[k] = v

    content["metrics"] = metrics
    return Message(
        content=content,
        reply_to=message
    )


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    synthesizer = get_synthesizer()
    log(
        f"{__name__}.evaluate:",
        (f"Successfully created synthesizer: {synthesizer}",),
        level=DEBUG
    )
    metrics = MetricRecord({"num-examples": 1})
    return Message(
        content=RecordDict({"metrics": metrics}),
        reply_to=message
    )
