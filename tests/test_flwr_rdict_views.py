from collections.abc import MutableMapping

import pytest
from flwr.app import ConfigRecord, MetricRecord, RecordDict

# noinspection PyProtectedMember
from fenris.flwr.rdict import RDictNamespaceView, _NamespaceView


def full_key(namespace: str, sub: str, sep: str) -> str:
    return f"{namespace}{sep}{sub}"


@pytest.fixture(params=["::", "|", "/", "."])
def sep(request) -> str:
    return request.param


def test_set_get_del_and_contains(sep: str):
    backing: MutableMapping[str, int] = {}
    view = _NamespaceView[int]("ns", backing, sep=sep)

    view["a"] = 1
    assert backing == {full_key("ns", "a", sep): 1}

    assert "a" in view
    assert view["a"] == 1

    del view["a"]
    assert full_key("ns", "a", sep) not in backing
    assert "a" not in view


def test_iter_len_items(sep: str):
    backing: MutableMapping[str, int] = {
        full_key("ns", "a", sep): 1,
        full_key("ns", "b", sep): 2,
        full_key("nsx", "c", sep): 3,
        full_key("other", "x", sep): 9,
    }
    view = _NamespaceView[int]("ns", backing, sep=sep)

    assert set(iter(view)) == {"a", "b"}
    assert len(view) == 2
    assert dict(view.items()) == {"a": 1, "b": 2}


def test_subkey_may_contain_sep(sep: str):
    backing: MutableMapping[str, int] = {}
    sub = f"a{sep}b"  # subkey contains the separator
    view = _NamespaceView[int]("ns", backing, sep=sep)

    view[sub] = 7
    assert full_key("ns", sub, sep) in backing
    assert view[sub] == 7
    assert sub in set(view)


def test_clear_removes_only_namespaced_keys(sep: str):
    backing: MutableMapping[str, int] = {
        full_key("ns", "a", sep): 1,
        full_key("ns", "b", sep): 2,
        full_key("other", "x", sep): 9,
    }
    view = _NamespaceView[int]("ns", backing, sep=sep)
    view.clear()
    assert backing == {full_key("other", "x", sep): 9}


def test_iterator_raises_on_mutation(sep: str):
    backing: MutableMapping[str, int] = {full_key("ns", "a", sep): 1}
    view = _NamespaceView[int]("ns", backing, sep=sep)

    it = iter(view)
    assert next(it) == "a"
    # Mutate backing during iteration
    backing[full_key("ns", "b", sep)] = 2
    with pytest.raises(RuntimeError):
        _ = next(it)


def test_write_through_isolated(sep: str):
    rd = RecordDict()
    view = RDictNamespaceView("ns", rd, sep=sep)

    view["cfg"] = ConfigRecord({"lr": 1e-3})
    view["loss"] = MetricRecord({"value": 0.5})

    assert full_key("ns", "cfg", sep) in rd.config_records
    assert full_key("ns", "loss", sep) in rd.metric_records

    other = RDictNamespaceView("other", rd, sep=sep)
    assert "cfg" not in other.config_records
    assert "loss" not in other.metric_records


def test_pass_not_rdict_raises():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        RDictNamespaceView(namespace="whatever", rdict={})
