from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import (
    Message,
    Context,
    RecordDict,
    ConfigRecord,
)
from pandas import DataFrame

from fedbench._flwr.serde import from_flwr, to_flwr
from fedbench.algorithms import registry as alg_registry
from fedbench.algorithms.algorithm import Synthesizer

app = ClientApp()
synthesizer_factory = None
na_protocol = None


def create_synthesizer(config: ConfigRecord | None = None) -> Synthesizer:
    global synthesizer_factory, na_protocol

    if synthesizer_factory is None:
        if config is None:
            raise ValueError("Can not load algorithm without config.")

        # noinspection PyUnnecessaryCast
        name: str = cast(str, config["algorithm-name"])
        algorithm = alg_registry.load(name)
        synthesizer_factory = algorithm.create_synthesizer
        na_protocol = algorithm.requires_non_array_protocol()

    return synthesizer_factory()


@app.query("init")
def init(message: Message, context: Context) -> Message:
    config = message.content.config_records["fedbench.config"]
    synthesizer = create_synthesizer(config)
    request = from_flwr(message)
    reply = synthesizer.init(request)

    return to_flwr(
        update=reply,
        reply_to=message,
        non_array_protocol=na_protocol
    )


@app.train()
def train(message: Message, context: Context) -> Message:
    config = message.content.config_records["fedbench.config"]
    synthesizer = create_synthesizer(config)
    request = from_flwr(message)
    reply = synthesizer.train(request, DataFrame())

    return to_flwr(
        update=reply,
        reply_to=message,
        non_array_protocol=na_protocol
    )


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    return Message(content=RecordDict(), reply_to=message)
