import random
from collections.abc import Callable

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from fedbench.core.payload import Payload
from fedbench.flwr.rdict import RDictNamespaceView
from fedbench.flwr.serde import FlwrSerde, Pickle

_RNG = np.random.default_rng(42)


@pytest.fixture(
    params=[
        pytest.param(False, id="pickle"),
        pytest.param(True, id="disable_pickle"),
    ]
)
def serde(request) -> FlwrSerde:
    return FlwrSerde(object_serde=Pickle(disabled=request.param))


@pytest.fixture
def make_random_ndarrays() -> Callable[[], list[np.ndarray]]:
    def factory():
        return list(_RNG.random((10, 10)) for _ in range(3))

    return factory


def test_empty(serde):
    orig = Payload()
    rdict = serde.to_flwr(orig)
    deserialized = serde.from_flwr(rdict)
    assert orig.is_empty()
    assert deserialized.is_empty()


def test_to_flwr_single_array_group(serde, make_random_ndarrays) -> None:
    update = Payload()
    orig = make_random_ndarrays()
    update.arrays["test-arrays"] = orig
    rdict = serde.to_flwr(update)
    retrieved = rdict["test-arrays"].to_numpy_ndarrays()
    for idx, arr in enumerate(orig):
        assert np.array_equal(arr, retrieved[idx]), "Arrays not equal"


def test_from_flwr_single_array_group(serde, make_random_ndarrays) -> None:
    orig = make_random_ndarrays()
    rdict = RecordDict({"test-arrays": ArrayRecord(orig)})
    update = serde.from_flwr(rdict)
    retrieved = update.arrays["test-arrays"]
    for idx, arr in enumerate(orig):
        assert np.array_equal(arr, retrieved[idx]), "Arrays not equal"


def test_to_flwr_multiple_array_groups(serde, make_random_ndarrays) -> None:
    update = Payload()
    orig1 = make_random_ndarrays()
    orig2 = make_random_ndarrays()
    update.arrays["test-arrays1"] = orig1
    update.arrays["test-arrays2"] = orig2

    rdict = serde.to_flwr(update)
    retrieved1 = rdict["test-arrays1"].to_numpy_ndarrays()
    retrieved2 = rdict["test-arrays2"].to_numpy_ndarrays()

    for idx, arr in enumerate(orig1):
        assert np.array_equal(arr, retrieved1[idx]), "Arrays not equal"

    for idx, arr in enumerate(orig2):
        assert np.array_equal(arr, retrieved2[idx]), "Arrays not equal"


def test_from_flwr_multiple_array_groups(serde, make_random_ndarrays) -> None:
    orig1 = make_random_ndarrays()
    orig2 = make_random_ndarrays()
    rdict = RecordDict(
        {
            "test-arrays1": ArrayRecord(orig1),
            "test-arrays2": ArrayRecord(orig2),
        }
    )
    update = serde.from_flwr(rdict)
    retrieved1 = update.arrays["test-arrays1"]
    retrieved2 = update.arrays["test-arrays2"]

    for idx, arr in enumerate(orig1):
        assert np.array_equal(arr, retrieved1[idx]), "Arrays not equal"

    for idx, arr in enumerate(orig2):
        assert np.array_equal(arr, retrieved2[idx]), "Arrays not equal"


def test_round_trip_combined(serde, make_random_ndarrays) -> None:
    """Update with arrays, metrics, and extras all populated simultaneously."""

    update = Payload()
    update.arrays["weights"] = make_random_ndarrays()
    update.metrics["train-metrics"] = {"loss": 0.42, "acc": 0.91}
    update.extras["meta"] = {"round": 3, "tag": "combined", "flag": True}

    rdict = serde.to_flwr(update)
    result = serde.from_flwr(rdict)

    for idx, arr in enumerate(update.arrays["weights"]):
        assert np.array_equal(arr, result.arrays["weights"][idx])
    assert result.metrics["train-metrics"] == update.metrics["train-metrics"]
    assert result.extras["meta"] == update.extras["meta"]


class PickleMe:
    def __init__(self, name):
        self.name = name
        self.desire = "I desperately want to be pickled!"
        self.some_dict = {
            "k": {"kk": {"kkk": 123}},
            "other_k": {"other_kk": {"other_kkk": 321}},
        }


def test_single_object_pickle():
    serde = FlwrSerde(object_serde=Pickle())
    update = Payload()
    orig = PickleMe("Some Name")
    update.objects["test-objects"] = {"pickle-me": orig}
    rdict = serde.to_flwr(update)
    deserialized = serde.from_flwr(rdict)
    unpickled = deserialized.objects["test-objects"]["pickle-me"]
    assert isinstance(unpickled, PickleMe), "Not a PickleMe instance"
    assert unpickled.name == orig.name
    assert unpickled.desire == orig.desire
    assert unpickled.some_dict == orig.some_dict


def test_metrics_single_group_all_types(serde):
    update = Payload()
    metrics = {
        "int": 1,
        "float": 1.1,
        "list[int]": [1, 2, 3],
        "list[float]": [1.1, 2.2, 3.3],
    }
    update.metrics["test-metrics"] = metrics
    rdict = serde.to_flwr(update)
    deserialized = serde.from_flwr(rdict)
    assert deserialized.metrics["test-metrics"] == metrics


def test_extras_single_group_all_types(serde):
    update = Payload()
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
    rdict = serde.to_flwr(update)
    deserialized = serde.from_flwr(rdict)
    assert deserialized.extras["test-extras"] == extras


def test_deserialize_rdict_view(serde, make_random_ndarrays):
    backing = RecordDict()

    backing["not-my-arrays"] = ArrayRecord(make_random_ndarrays())
    backing["not-my-config"] = ConfigRecord({"hello": "not-my-world"})
    backing["not-my-metrics"] = MetricRecord({"not-my-value": 1.2})

    view = RDictNamespaceView("my-awesome-space", backing)
    view["arrays"] = ArrayRecord(make_random_ndarrays())
    view["config"] = ConfigRecord({"hello": "world"})
    view["metrics"] = MetricRecord({"value": 1.0})
    update = serde.from_flwr(view)

    assert "arrays" in update.arrays
    assert "config" in update.extras
    assert "metrics" in update.metrics

    assert "not-my-arrays" not in update.arrays
    assert "not-my-config" not in update.extras
    assert "not-my-metrics" not in update.metrics


def test_disable_pickle_raises():
    serde = FlwrSerde(Pickle(disabled=True))
    update = Payload()
    update.objects["test-objects"] = {"pickle-me": None}
    with pytest.raises(TypeError):
        serde.to_flwr(update)
