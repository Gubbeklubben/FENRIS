from collections.abc import Callable
import random

import numpy as np
import pytest
from flwr.common import Message, RecordDict, ArrayRecord

# noinspection PyProtectedMember
from fedbench._flwr.serde import to_flwr_message, from_flwr_message
from fedbench.common import MessageContent

_RNG = np.random.default_rng(42)


def test_to_flwr_empty():
    msg_content = MessageContent()
    flwr_message = to_flwr_message(
        msg_content,
        message_type="train",
        dst_node_id=1
    )
    assert not flwr_message.content, ("Empty MessageContent -> non-empty flwr "
                                      "Message")


def test_from_flwr_empty():
    flwr_message = Message(
        message_type="train",
        dst_node_id=1,
        content=RecordDict()
    )
    msg_content = from_flwr_message(flwr_message)
    assert msg_content.is_empty(), ("Empty flwr Message -> non-empty "
                                    "MessageContent")


@pytest.fixture
def make_random_ndarrays() -> Callable[[], list[np.ndarray]]:
    def factory():
        return list(_RNG.random((10, 10)) for _ in range(3))
    return factory


def test_to_flwr_single_array_group(make_random_ndarrays) -> None:
    msg_content = MessageContent()
    orig = make_random_ndarrays()
    msg_content.add_arrays("test-arrays", orig)
    flwr_message = to_flwr_message(
        msg_content,
        message_type="train",
        dst_node_id=1
    )
    retrieved = flwr_message.content["test-arrays"].to_numpy_ndarrays()
    for idx, arr in enumerate(orig):
        assert np.array_equal(arr, retrieved[idx]), "Arrays not equal"


def test_from_flwr_single_array_group(make_random_ndarrays) -> None:
    orig = make_random_ndarrays()
    flwr_message = Message(
        message_type="train",
        dst_node_id=1,
        content=RecordDict({"test-arrays": ArrayRecord(orig)})
    )
    msg_content = from_flwr_message(flwr_message)
    retrieved = msg_content.arrays["test-arrays"]
    for idx, arr in enumerate(orig):
        assert np.array_equal(arr, retrieved[idx]), "Arrays not equal"


def test_to_flwr_multiple_array_groups(make_random_ndarrays) -> None:
    msg_content = MessageContent()
    orig1 = make_random_ndarrays()
    orig2 = make_random_ndarrays()
    msg_content.add_arrays("test-arrays1", orig1)
    msg_content.add_arrays("test-arrays2", orig2)
    flwr_message = to_flwr_message(
        msg_content,
        message_type="train",
        dst_node_id=1
    )
    retrieved1 = flwr_message.content["test-arrays1"].to_numpy_ndarrays()
    retrieved2 = flwr_message.content["test-arrays2"].to_numpy_ndarrays()

    for idx, arr in enumerate(orig1):
        assert np.array_equal(arr, retrieved1[idx]), "Arrays not equal"

    for idx, arr in enumerate(orig2):
        assert np.array_equal(arr, retrieved2[idx]), "Arrays not equal"


def test_from_flwr_multiple_array_groups(make_random_ndarrays) -> None:
    orig1 = make_random_ndarrays()
    orig2 = make_random_ndarrays()
    flwr_message = Message(
        message_type="train",
        dst_node_id=1,
        content=RecordDict({
            "test-arrays1": ArrayRecord(orig1),
            "test-arrays2": ArrayRecord(orig2),
        })
    )
    msg_content = from_flwr_message(flwr_message)
    retrieved1 = msg_content.arrays["test-arrays1"]
    retrieved2 = msg_content.arrays["test-arrays2"]

    for idx, arr in enumerate(orig1):
        assert np.array_equal(arr, retrieved1[idx]), "Arrays not equal"

    for idx, arr in enumerate(orig2):
        assert np.array_equal(arr, retrieved2[idx]), "Arrays not equal"


class PickleMe:
    def __init__(self, name):
        self.name = name
        self.desire = "I desperately want to be pickled!"
        self.some_dict = {
            "k": {"kk": {"kkk": 123}},
            "other_k": {"other_kk": {"other_kkk": 321}}
        }


def test_single_object_pickle():
    msg_content = MessageContent()
    orig = PickleMe("Some Name")
    msg_content.add_objects("test-objects", {"pickle-me": orig})
    flwr_message = to_flwr_message(
        msg_content,
        message_type="train",
        dst_node_id=1,
        non_array_protocol="pickle",
        allow_pickle=True
    )
    decoded_content = from_flwr_message(
        flwr_message,
        allow_pickle=True
    )
    unpickled = decoded_content.objects["test-objects"]["pickle-me"]
    assert isinstance(unpickled, PickleMe), "Not a PickleMe instance"
    assert unpickled.name == orig.name
    assert unpickled.desire == orig.desire
    assert unpickled.some_dict == orig.some_dict


def test_metrics_single_group_all_types():
    msg_content = MessageContent()
    metrics = {
        "int": 1,
        "float": 1.1,
        "list[int]": [1, 2, 3],
        "list[float]": [1.1, 2.2, 3.3],
    }
    msg_content.add_metrics("test-metrics", metrics)
    flwr_message = to_flwr_message(
        msg_content,
        message_type="train",
        dst_node_id=1,
    )
    decoded_content = from_flwr_message(flwr_message)
    assert decoded_content.metrics["test-metrics"] == metrics


def test_config_single_group_all_types():
    msg_content = MessageContent()
    config = {
        "int": 1,
        "float": 1.1,
        "list[int]": [1, 2, 3],
        "list[float]": [1.1, 2.2, 3.3],
        "bytes": random.randbytes(100),
        "list[bytes]": [random.randbytes(8) for _ in range(3)],
        "bool": False,
        "list[bool]": [True, False, True],
        "str": "Hello!",
        "list[str]": ["1", "2", "3"],
    }
    msg_content.add_config("test-config", config)
    flwr_message = to_flwr_message(
        msg_content,
        message_type="train",
        dst_node_id=1,
    )
    decoded_content = from_flwr_message(flwr_message)
    assert decoded_content.config["test-config"] == config


