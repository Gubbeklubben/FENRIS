"""Tests for the fed_chaos algorithm."""

import pandas as pd
import pytest

from fedbench.algorithms.fed_chaos import (
    FedChaos,
    FedChaosCoordinator,
    FedChaosSynthesizer,
    _ChaosConfig,
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
) -> _ChaosConfig:
    return _ChaosConfig(scenario=scenario, point=point, exception=exception)


# ── Config validation ────────────────────────────────────────────────


class TestChaosConfig:
    def test_valid_config(self) -> None:
        cfg = _make_config("crash", "synth_train")
        assert cfg.scenario == "crash"
        assert cfg.point == "synth_train"

    def test_invalid_scenario(self) -> None:
        with pytest.raises(ValueError, match="Unknown chaos scenario"):
            _make_config(scenario="nonexistent")

    def test_invalid_exception(self) -> None:
        with pytest.raises(ValueError, match="Unknown exception"):
            _make_config(scenario="crash", exception="FakeError")


# ── Component smoke tests ───────────────────────────────────────────


class TestFedChaos:
    def test_global_init_crash(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="crash", point="global_init")
        with pytest.raises(RuntimeError, match="Chaos:.*global_init"):
            algo.global_init(seed=42, schema=schema, dataset=dataset)

    def test_coordinator_crash(self) -> None:
        coord = FedChaosCoordinator(_make_config(scenario="crash", point="coord_train"))
        with pytest.raises(RuntimeError, match="Chaos:"):
            coord.aggregate_train([(0, Update())])

    def test_synthesizer_crash(self) -> None:
        synth = FedChaosSynthesizer(_make_config(scenario="crash", point="synth_train"))
        with pytest.raises(RuntimeError, match="Chaos:.*synth_train"):
            synth.train(Update(), pd.DataFrame({"a": [1]}))

    def test_no_trigger_when_point_differs(self) -> None:
        synth = FedChaosSynthesizer(_make_config(point="global_init"))
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    def test_fed_chaos_registered(self) -> None:
        from fedbench.algorithms import register_builtin_algorithms
        from fedbench.core.algorithm import Algorithm
        from fedbench.runtime.registry import FactoryRegistry

        registry: FactoryRegistry[Algorithm] = FactoryRegistry(
            group="fedbench.test.algorithms",
            product_cls=Algorithm,  # type: ignore[type-abstract]
        )
        register_builtin_algorithms(registry)
        assert "fed_chaos" in registry
