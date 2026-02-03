from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict,
    ArrayRecord,
    MetricRecord,
    Array,
)

from fedbench._plugins import load_synthesizer_factory
from fedbench.common import InitRequest

app = ClientApp()
_synthesizer_factory = None


@app.query("init")
def init(message: Message, context: Context) -> Message:
    config = message.content.config_records["config"]
    algorithm_name = str(config["algorithm-name"])

    global _synthesizer_factory
    _synthesizer_factory = load_synthesizer_factory(algorithm_name)
    synthesizer = _synthesizer_factory()

    init_response = synthesizer.init(
        InitRequest(message.metadata.dst_node_id, None)
    )
    if init_response.statistics is None:
        return Message(
            content=RecordDict(),
            reply_to=message
        )
    statistics = ArrayRecord(
        {k: Array(ndarray) for k, ndarray in init_response.statistics.items()}
    )
    return Message(
        content=RecordDict({"init": statistics}),
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
    arrays = ArrayRecord()
    metrics = MetricRecord({"num-examples": 1})
    return Message(
        content=RecordDict({"arrays": arrays, "metrics": metrics}),
        reply_to=message
    )


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    metrics = MetricRecord({"num-examples": 1})
    return Message(
        content=RecordDict({"metrics": metrics}),
        reply_to=message
    )
