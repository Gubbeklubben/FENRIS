"""Tests for the fed_naughty algorithm."""

import pandas as pd
import pytest

# noinspection PyProtectedMember
from fedbench.algorithms.fed_naughty import (
    FedNaughty,
    FedNaughtyCoordinator,
    FedNaughtySynthesizer,
    _NaughtyConfig,
)
from fedbench.core.data.schemas import ColumnSchema, TableSchema
from fedbench.core.update import Update

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def schema() -> TableSchema:
    return TableSchema(
        columns=(
            ColumnSchema("age", "continuous"),
            ColumnSchema("label", "binary"),
        )
    )


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame({"age": [25, 30, 45], "label": [0, 1, 0]})


def _make_config(
    scenario: str = "crash",
    point: str = "synth_train",
    exception: str = "RuntimeError",
) -> _NaughtyConfig:
    return _NaughtyConfig(scenario=scenario, point=point, exception=exception)


# ── Config validation ────────────────────────────────────────────────


class TestNaughtyConfig:
    def test_valid_config(self) -> None:
        cfg = _make_config("crash", "synth_train")
        assert cfg.scenario == "crash"
        assert cfg.point == "synth_train"

    def test_invalid_scenario(self) -> None:
        with pytest.raises(ValueError):
            _make_config(scenario="nonexistent")

    def test_invalid_exception(self) -> None:
        with pytest.raises(ValueError, match="Unknown exception"):
            _make_config(scenario="crash", exception="FakeError")


# ── Component smoke tests ───────────────────────────────────────────


class TestFedChaos:
    def test_global_init_crash(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedNaughty(scenario="crash", point="global_init", exception="TypeError")
        with pytest.raises(TypeError):
            algo.global_init(seed=42, schema=schema, dataset=dataset)

    def test_coordinator_crash(self) -> None:
        coord = FedNaughtyCoordinator(
            _make_config(scenario="crash", point="coord_train")
        )
        with pytest.raises(Exception):
            coord.aggregate_train([(0, Update())])

    def test_synthesizer_crash(self) -> None:
        synth = FedNaughtySynthesizer(
            _make_config(scenario="crash", point="synth_train")
        )
        with pytest.raises(Exception):
            synth.train(Update(), pd.DataFrame({"a": [1]}))

    def test_no_trigger_when_point_differs(self) -> None:
        synth = FedNaughtySynthesizer(_make_config(point="global_init"))
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
