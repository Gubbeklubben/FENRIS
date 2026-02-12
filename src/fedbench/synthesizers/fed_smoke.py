from collections.abc import Iterable
from typing import Any

from pandas import DataFrame

from fedbench.common import Arrays, MessageContent
from fedbench.common import log_calls
from fedbench.synthesizers.synthesizer import ClientComponent
from fedbench.synthesizers.synthesizer import ServerComponent, Synthesizer


class FedSmoke(Synthesizer):
    @property
    def non_array_protocols(self) -> tuple[str, ...]:
        return ("msgpack",)

    def server_factory(self) -> ServerComponent:
        return FedSmokeServer()

    def client_factory(self) -> ClientComponent:
        return FedSmokeClient()


class FedSmokeServer(ServerComponent):
    @log_calls(__name__)
    def aggregate_train(
            self,
            replies: Iterable[MessageContent]) -> tuple[Arrays | None, Any | None]:

        reply = None
        for reply in replies:
            pass

        if reply is not None:
            return None, reply.objects

        return None, None



class FedSmokeClient(ClientComponent):
    def __init__(self):
        self._bullshit_state = {
            "bs1": {"bs11": 1, "bs12": 2, "bs13": 3, "nested_dict": {1: 1}},
            "bs2": {"bs21": 1, "bs22": 2, "bs23": 3, "nested_dict": {2: 2}},
        }

    @log_calls(__name__)
    def train(
            self,
            request: MessageContent,
            data: DataFrame) -> MessageContent:

        msg_content = MessageContent()
        msg_content.add_object("my-bs-state", self._bullshit_state)
        return msg_content

    @log_calls(__name__)
    def sample(
            self,
            request: MessageContent,
            num_rows: int,
            seed: int) -> DataFrame:

        num_records = 10
        data = {
            "ints": np.random.randint(0, 100, size=num_records),
            "floats": np.random.randn(num_records),
            "dates": pd.date_range('2023-01-01', periods=num_records),
            "categories": np.random.choice(["a", "b", "c"], size=num_records)
        }
        return DataFrame(data)