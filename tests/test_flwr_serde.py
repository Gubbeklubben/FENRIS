import random
from collections.abc import Callable
from typing import Iterable

import numpy as np
import pytest
from flwr.common import Message, RecordDict, ArrayRecord

from fedbench.core.update import Update
from fedbench.flwr.serde import (
    FlwrSerializer, FlwrDeserializer,
    to_flwr_pickle, from_flwr_pickle, to_flwr_disable_pickle
)

_RNG = np.random.default_rng(42)


@pytest.fixture
def serde() -> Iterable[tuple[FlwrSerializer, FlwrDeserializer]]:
    return (
        (to_flwr_pickle, from_flwr_pickle),
        (to_flwr_disable_pickle, from_flwr_pickle)
    )


@pytest.fixture
def make_random_ndarrays() -> Callable[[], list[np.ndarray]]:
    def factory():
        return list(_RNG.random((10, 10)) for _ in range(3))
    return factory


def test_empty(serde):
    orig = Update()
    for to_flwr, from_flwr in serde:
        flwr_message = to_flwr(
            orig,
            message_type="train",
            dst_node_id=1
        )
        deserialized = from_flwr(flwr_message)
        assert orig.is_empty()
        assert deserialized.is_empty()


def test_to_flwr_single_array_group(
        serde,
        make_random_ndarrays) -> None:
    update = Update()
    orig = make_random_ndarrays()
    update.arrays["test-arrays"] = orig
    for to_flwr, _ in serde:
        flwr_message = to_flwr(
            update,
            message_type="train",
            dst_node_id=1
        )
        retrieved = flwr_message.content["test-arrays"].to_numpy_ndarrays()
        for idx, arr in enumerate(orig):
            assert np.array_equal(arr, retrieved[idx]), "Arrays not equal"


def test_from_flwr_single_array_group(
        serde,
        make_random_ndarrays) -> None:

    orig = make_random_ndarrays()
    for _, from_flwr in serde:
        flwr_message = Message(
            message_type="train",
            dst_node_id=1,
            content=RecordDict({"test-arrays": ArrayRecord(orig)})
        )
        update = from_flwr(flwr_message)
        retrieved = update.arrays["test-arrays"]
        for idx, arr in enumerate(orig):
            assert np.array_equal(arr, retrieved[idx]), "Arrays not equal"


def test_to_flwr_multiple_array_groups(
        serde,
        make_random_ndarrays) -> None:

    update = Update()
    orig1 = make_random_ndarrays()
    orig2 = make_random_ndarrays()
    update.arrays["test-arrays1"] = orig1
    update.arrays["test-arrays2"] = orig2

    for to_flwr, _ in serde:
        flwr_message = to_flwr(
            update,
            message_type="train",
            dst_node_id=1
        )
        retrieved1 = flwr_message.content["test-arrays1"].to_numpy_ndarrays()
        retrieved2 = flwr_message.content["test-arrays2"].to_numpy_ndarrays()

        for idx, arr in enumerate(orig1):
            assert np.array_equal(arr, retrieved1[idx]), "Arrays not equal"

        for idx, arr in enumerate(orig2):
            assert np.array_equal(arr, retrieved2[idx]), "Arrays not equal"


def test_from_flwr_multiple_array_groups(
        serde,
        make_random_ndarrays) -> None:

    orig1 = make_random_ndarrays()
    orig2 = make_random_ndarrays()
    for _, from_flwr in serde:
        flwr_message = Message(
            message_type="train",
            dst_node_id=1,
            content=RecordDict({
                "test-arrays1": ArrayRecord(orig1),
                "test-arrays2": ArrayRecord(orig2),
            })
        )
        update = from_flwr(flwr_message)
        retrieved1 = update.arrays["test-arrays1"]
        retrieved2 = update.arrays["test-arrays2"]

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
    update = Update()
    orig = PickleMe("Some Name")
    update.objects["test-objects"] = {"pickle-me": orig}
    flwr_message = to_flwr_pickle(
        update,
        message_type="train",
        dst_node_id=1
    )
    deserialized = from_flwr_pickle(flwr_message)
    unpickled = deserialized.objects["test-objects"]["pickle-me"]
    assert isinstance(unpickled, PickleMe), "Not a PickleMe instance"
    assert unpickled.name == orig.name
    assert unpickled.desire == orig.desire
    assert unpickled.some_dict == orig.some_dict


def test_metrics_single_group_all_types(serde):
    for to_flwr, from_flwr in serde:
        update = Update()
        metrics = {
            "int": 1,
            "float": 1.1,
            "list[int]": [1, 2, 3],
            "list[float]": [1.1, 2.2, 3.3],
        }
        update.metrics["test-metrics"] = metrics
        flwr_message = to_flwr(
            update,
            message_type="train",
            dst_node_id=1,
        )
        deserialized = from_flwr(flwr_message)
        assert deserialized.metrics["test-metrics"] == metrics


def test_extras_single_group_all_types(serde):
    for to_flwr, from_flwr in serde:
        update = Update()
        extras = {
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
        update.extras["test-extras"] = extras
        flwr_message = to_flwr(
            update,
            message_type="train",
            dst_node_id=1,
        )
        deserialized = from_flwr(flwr_message)
        assert deserialized.extras["test-extras"] == extras


def test_disable_pickle_raises():
    update = Update()
    update.objects["test-objects"] = {"pickle-me": None}
    with pytest.raises(RuntimeError):
        to_flwr_disable_pickle(
            update,
            message_type="train",
            dst_node_id=1
        )
