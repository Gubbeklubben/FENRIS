import pytest

from fenris.app.plugins import Registry
from fenris.core.algorithm import Synthesizer
from tests.fake_components import (
    abstract_synth_entry_points,
    not_a_class_entry_points,
    not_a_synth_entry_points,
)


@pytest.fixture(scope="function")
def not_a_synth_registry(monkeypatch):
    monkeypatch.setattr("fenris.app.plugins.entry_points", not_a_synth_entry_points)
    return Registry("fenris.synthesizers", Synthesizer)


@pytest.fixture(scope="function")
def abstract_synth_registry(monkeypatch):
    monkeypatch.setattr("fenris.app.plugins.entry_points", abstract_synth_entry_points)
    return Registry("fenris.synthesizers", Synthesizer)


@pytest.fixture(scope="function")
def not_a_class_registry(monkeypatch):
    monkeypatch.setattr("fenris.app.plugins.entry_points", not_a_class_entry_points)
    return Registry("fenris.synthesizers", Synthesizer)


def test_raises_on_not_subclass(not_a_synth_registry):
    with pytest.raises(TypeError, match="not a subclass"):
        not_a_synth_registry.load("not_a_synthesizer")


def test_raises_on_abstract(abstract_synth_registry):
    with pytest.raises(TypeError, match="is abstract"):
        abstract_synth_registry.load("abstract_synthesizer")


def test_raises_on_not_class(not_a_class_registry):
    with pytest.raises(TypeError, match="not a class"):
        not_a_class_registry.load("not_a_class")
