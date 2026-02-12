from logging import DEBUG
from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict,
    MetricRecord,
    ConfigRecord,
)
from pandas import DataFrame

from fedbench._flwr.serde import from_flwr_message, to_flwr_message
from fedbench.synthesizers import load_factory as load_algorithm_factory
from fedbench.synthesizers.synthesizer import ClientComponent

app = ClientApp()
synthesizer_factory = None


def get_synthesizer(config: ConfigRecord | None = None) -> ClientComponent:
    global synthesizer_factory

    if synthesizer_factory is None:
        if config is None:
            raise ValueError("Can not load synthesizer without config.")

        # noinspection PyUnnecessaryCast
        algorithm_name: str = cast(str, config["algorithm-name"])
        algorithm_factory = load_algorithm_factory(algorithm_name)
        algorithm = algorithm_factory()
        synthesizer_factory = algorithm.synthesizer_factory

    return synthesizer_factory()


@app.query("init")
def init(message: Message, context: Context) -> Message:
    config = message.content.config_records["config"]
    synthesizer = get_synthesizer(config)
    request = decode(message)
    response = synthesizer.init(request)
    return to_flwr_message(response, message_type="init", reply_to=message)

@app.train()
def train(message: Message, context: Context) -> Message:
    # Load data
    # Call synthesizer factory
    # Call synthesizer.train
    # Get synthesizer weights / other relevant stuff
    # Convert to Flower Message and return it
    synthesizer = get_synthesizer()
    request = decode(message)
    response = synthesizer.train(request, DataFrame())
    return to_flwr_message(response, message_type="train", reply_to=message)


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
