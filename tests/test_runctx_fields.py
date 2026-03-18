from collections.abc import Iterable

import pytest

# noinspection PyProtectedMember
from fedbench.runtime.runcontext import RunContext, _RunCtxField


@pytest.fixture
def instance() -> RunContext:
    # noinspection PyTypeChecker
    return RunContext(
        "test",
        config=None,
        eventbus=None,
        scalability_collector=None,
    )


@pytest.fixture
def fields() -> Iterable[str]:
    return (k for k, v in RunContext.__dict__.items() if isinstance(v, _RunCtxField))


def test_get_before_set_raises(instance, fields: Iterable[str]):
    for name in fields:
        with pytest.raises(AttributeError):
            getattr(instance, name)


def test_set_twice_raises(instance, fields: Iterable[str]):
    for name in fields:
        setattr(instance, name, object())
        with pytest.raises(RuntimeError):
            setattr(instance, name, object())


def test_get_after_set_returns_expected(instance, fields: Iterable[str]):
    for name in fields:
        value = object()
        setattr(instance, name, value)
        assert getattr(instance, name) is value
