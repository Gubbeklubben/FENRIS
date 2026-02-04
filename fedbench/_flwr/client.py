from logging import DEBUG
from typing import cast

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
from fedbench.common import InitRequest, log
from fedbench.synthesizer import Synthesizer

app = ClientApp()
_synthesizer_factory = None


def _get_synthesizer(config: ConfigRecord) -> Synthesizer:
    global _synthesizer_factory
    if _synthesizer_factory is None:
        # noinspection PyUnnecessaryCast
        algorithm_name: str = cast(str, config["algorithm-name"])
        _synthesizer_factory = load_synthesizer_factory(algorithm_name)
    return _synthesizer_factory()


@app.query("init")
def init(message: Message, context: Context) -> Message:
    config = message.content.config_records["config"]
    synthesizer = _get_synthesizer(config)
    init_response = synthesizer.init(
        InitRequest(message.metadata.dst_node_id, None)
    )
    content = RecordDict()
    if init_response.statistics is not None:
        record = ArrayRecord(
            {k: Array(ndarray)
            for k, ndarray in init_response.statistics.items()}
        )
        content["init"] = record

    return Message(
        content=content,
        reply_to=message
    )

@app.train()
def train(message: Message, context: Context) -> Message:
    # Load data
    # Call synthesizer factory
    # Set synthesizer weights from converted message content
    # Call synthesizer.train
    # Get synthesizer weights / other relevant stuff
    # Convert to Flower Message and return it
    config = message.content.config_records["config"]
    synthesizer = _get_synthesizer(config)
    log(
        f"{__name__}.train:",
        (f"Successfully created synthesizer: {synthesizer}",),
        level=DEBUG
    )
    arrays = ArrayRecord()
    metrics = MetricRecord({"num-examples": 1})
    return Message(
        content=RecordDict({"arrays": arrays, "metrics": metrics}),
        reply_to=message
    )


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    config = message.content.config_records["config"]
    synthesizer = _get_synthesizer(config)
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
