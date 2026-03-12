"""Tests for the fed_chaos algorithm."""

import math
from unittest.mock import patch

import pandas as pd
import pytest

from fedbench.algorithms.fed_chaos import (
    _VALID_POINTS,
    _VALID_SCENARIOS,
    FedChaos,
    FedChaosCoordinator,
    FedChaosSynthesizer,
    _ChaosConfig,
    _leak_store,
)
from fedbench.core.algorithm import GlobalInitArtifacts
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
    intensity: float = 1.0,
    exception: str = "RuntimeError",
) -> _ChaosConfig:
    return _ChaosConfig(
        scenario=scenario, point=point, intensity=intensity, exception=exception
    )


# ── Config validation ────────────────────────────────────────────────


class TestChaosConfig:
    def test_valid_config(self) -> None:
        cfg = _make_config("crash", "synth_train")
        assert cfg.scenario == "crash"
        assert cfg.point == "synth_train"

    def test_invalid_scenario(self) -> None:
        with pytest.raises(ValueError, match="Unknown chaos scenario"):
            _make_config(scenario="nonexistent")

    def test_invalid_point(self) -> None:
        with pytest.raises(ValueError, match="Unknown chaos point"):
            _make_config(point="nonexistent")

    def test_invalid_exception(self) -> None:
        with pytest.raises(ValueError, match="Unknown exception"):
            _make_config(scenario="crash", exception="FakeError")

    def test_exception_not_validated_for_non_crash(self) -> None:
        # exception param is only validated when scenario == "crash"
        cfg = _make_config(scenario="delay", exception="FakeError")
        assert cfg.exception == "FakeError"

    @pytest.mark.parametrize("scenario", sorted(_VALID_SCENARIOS))
    def test_all_scenarios_valid(self, scenario: str) -> None:
        cfg = _make_config(scenario=scenario)
        assert cfg.scenario == scenario

    @pytest.mark.parametrize("point", sorted(_VALID_POINTS))
    def test_all_points_valid(self, point: str) -> None:
        cfg = _make_config(point=point)
        assert cfg.point == point


# ── FedChaos Algorithm ───────────────────────────────────────────────


class TestFedChaos:
    def test_default_construction(self) -> None:
        algo = FedChaos()
        assert algo._config.scenario == "crash"
        assert algo._config.point == "synth_train"

    def test_custom_construction(self) -> None:
        algo = FedChaos(scenario="delay", point="global_init", intensity=5.0)
        assert algo._config.scenario == "delay"
        assert algo._config.point == "global_init"
        assert algo._config.intensity == 5.0

    def test_coordinator_spec_produces_coordinator(self) -> None:
        algo = FedChaos()
        coord = algo.coordinator_spec.factory()
        assert isinstance(coord, FedChaosCoordinator)

    def test_synthesizer_spec_produces_synthesizer(self) -> None:
        algo = FedChaos()
        synth = algo.synthesizer_spec.factory()
        assert isinstance(synth, FedChaosSynthesizer)

    def test_global_init_no_trigger(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="crash", point="synth_train")
        result = algo.global_init(seed=42, schema=schema, dataset=dataset)
        assert result is None

    def test_global_init_crash(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="crash", point="global_init")
        with pytest.raises(RuntimeError, match="Chaos:.*global_init"):
            algo.global_init(seed=42, schema=schema, dataset=dataset)

    def test_global_init_corrupt(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="corrupt", point="global_init")
        result = algo.global_init(seed=42, schema=schema, dataset=dataset)
        assert isinstance(result, GlobalInitArtifacts)
        assert "corrupt-nan" in result.coordinator.arrays
        assert math.isnan(result.coordinator.metrics["corrupt"]["nan"])

    def test_global_init_empty(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="empty", point="global_init")
        result = algo.global_init(seed=42, schema=schema, dataset=dataset)
        assert isinstance(result, GlobalInitArtifacts)
        assert result.coordinator.is_empty()
        assert result.synthesizer.is_empty()

    def test_global_init_wrong_type(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="wrong_type", point="global_init")
        result = algo.global_init(seed=42, schema=schema, dataset=dataset)
        assert result == "NOT_ARTIFACTS"

    def test_global_init_large_payload(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="large_payload", point="global_init", intensity=0.01)
        with pytest.raises(MemoryError, match="Chaos: payload too large"):
            algo.global_init(seed=42, schema=schema, dataset=dataset)

    def test_global_init_delay(
        self, schema: TableSchema, dataset: pd.DataFrame
    ) -> None:
        algo = FedChaos(scenario="delay", point="global_init", intensity=0.0)
        with patch("fedbench.algorithms.fed_chaos.time.sleep") as mock_sleep:
            result = algo.global_init(seed=42, schema=schema, dataset=dataset)
            mock_sleep.assert_called_once_with(0.0)
        assert result is None

    def test_global_init_leak(self, schema: TableSchema, dataset: pd.DataFrame) -> None:
        initial_count = len(_leak_store)
        algo = FedChaos(scenario="leak", point="global_init", intensity=0.001)
        algo.global_init(seed=42, schema=schema, dataset=dataset)
        assert len(_leak_store) == initial_count + 1


# ── Coordinator ──────────────────────────────────────────────────────


class TestFedChaosCoordinator:
    def test_global_state_initially_empty(self) -> None:
        coord = FedChaosCoordinator(_make_config(point="synth_train"))
        assert isinstance(coord.global_state, Update)
        assert coord.global_state.is_empty()

    def test_configure_fed_init_yields_updates(self) -> None:
        coord = FedChaosCoordinator(_make_config(point="synth_train"))
        schema = TableSchema()
        result = list(coord.configure_fed_init(42, schema, [0, 1, 2]))
        assert len(result) == 3
        assert all(isinstance(u, Update) for _, u in result)

    def test_configure_fed_init_crash(self) -> None:
        coord = FedChaosCoordinator(_make_config(point="coord_fed_init"))
        schema = TableSchema()
        with pytest.raises(RuntimeError, match="Chaos:"):
            # Must consume the generator to trigger the crash.
            list(coord.configure_fed_init(42, schema, [0]))

    def test_aggregate_train_stores_state(self) -> None:
        coord = FedChaosCoordinator(_make_config(point="synth_train"))
        reply = Update(metrics={"m": {"val": 1.0}})
        coord.aggregate_train([(0, reply)])
        assert coord.global_state is reply

    def test_aggregate_train_crash(self) -> None:
        coord = FedChaosCoordinator(_make_config(scenario="crash", point="coord_train"))
        with pytest.raises(RuntimeError, match="Chaos:"):
            coord.aggregate_train([(0, Update())])

    def test_aggregate_train_corrupt(self) -> None:
        coord = FedChaosCoordinator(
            _make_config(scenario="corrupt", point="coord_train")
        )
        coord.aggregate_train([(0, Update())])
        state = coord.global_state
        assert state is not None
        assert "corrupt-nan" in state.arrays

    def test_aggregate_train_empty(self) -> None:
        coord = FedChaosCoordinator(_make_config(scenario="empty", point="coord_train"))
        coord.aggregate_train([(0, Update(metrics={"m": {"val": 1.0}}))])
        state = coord.global_state
        assert state is not None
        assert state.is_empty()

    def test_aggregate_train_wrong_type_leaves_state_unchanged(self) -> None:
        coord = FedChaosCoordinator(
            _make_config(scenario="wrong_type", point="coord_train")
        )
        coord.aggregate_train([(0, Update())])
        # wrong_type returns early without updating _state, so it keeps its
        # initial empty Update value — avoid corrupting state with a wrong type.
        assert isinstance(coord.global_state, Update)

    def test_aggregate_train_large_payload(self) -> None:
        coord = FedChaosCoordinator(
            _make_config(scenario="large_payload", point="coord_train", intensity=0.01)
        )
        with pytest.raises(MemoryError, match="Chaos: payload too large"):
            coord.aggregate_train([(0, Update())])

    def test_train_generator_normal_flow(self) -> None:
        coord = FedChaosCoordinator(_make_config(point="synth_train"))
        # Pre-set state so configure_train works.
        coord._state = Update(metrics={"m": {"v": 1.0}})

        gen = coord.train([0, 1])
        batch = next(gen)
        batch = list(batch)
        assert len(batch) == 2

        reply = Update(metrics={"m": {"v": 2.0}})
        with pytest.raises(StopIteration):
            gen.send([(0, reply), (1, reply)])

    def test_train_infinite_loop(self) -> None:
        coord = FedChaosCoordinator(
            _make_config(scenario="infinite_loop", point="coord_train")
        )
        gen = coord.train([0, 1])

        # Run a few rounds to verify it doesn't stop.
        for i in range(5):
            batch = next(gen) if i == 0 else gen.send([(0, Update()), (1, Update())])
            batch = list(batch)
            assert len(batch) == 2

        gen.close()

    def test_coord_delay(self) -> None:
        coord = FedChaosCoordinator(
            _make_config(scenario="delay", point="coord_train", intensity=0.0)
        )
        with patch("fedbench.algorithms.fed_chaos.time.sleep") as mock_sleep:
            coord.aggregate_train([(0, Update())])
            mock_sleep.assert_called_once_with(0.0)


# ── Synthesizer ──────────────────────────────────────────────────────


class TestFedChaosSynthesizer:
    def test_train_no_trigger(self) -> None:
        synth = FedChaosSynthesizer(_make_config(point="global_init"))
        data = pd.DataFrame({"a": [1, 2, 3]})
        result = synth.train(Update(), data)
        assert isinstance(result, Update)
        assert result.metrics["chaos"]["rows"] == 3
        assert result.metrics["chaos"]["round"] == 1

    def test_train_crash(self) -> None:
        synth = FedChaosSynthesizer(_make_config(scenario="crash", point="synth_train"))
        with pytest.raises(RuntimeError, match="Chaos:.*synth_train"):
            synth.train(Update(), pd.DataFrame({"a": [1]}))

    def test_train_corrupt(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="corrupt", point="synth_train")
        )
        result = synth.train(Update(), pd.DataFrame({"a": [1]}))
        assert isinstance(result, Update)
        assert "corrupt-nan" in result.arrays

    def test_train_wrong_type(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="wrong_type", point="synth_train")
        )
        result = synth.train(Update(), pd.DataFrame({"a": [1]}))
        assert result == "THIS_IS_NOT_AN_UPDATE"

    def test_train_empty(self) -> None:
        synth = FedChaosSynthesizer(_make_config(scenario="empty", point="synth_train"))
        result = synth.train(Update(), pd.DataFrame({"a": [1]}))
        assert isinstance(result, Update)
        assert result.is_empty()

    def test_train_large_payload(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="large_payload", point="synth_train", intensity=0.01)
        )
        with pytest.raises(MemoryError, match="Chaos: payload too large"):
            synth.train(Update(), pd.DataFrame({"a": [1]}))

    def test_train_call_count_increments(self) -> None:
        synth = FedChaosSynthesizer(_make_config(point="global_init"))
        data = pd.DataFrame({"a": [1]})
        synth.train(Update(), data)
        synth.train(Update(), data)
        result = synth.train(Update(), data)
        assert result.metrics["chaos"]["round"] == 3

    def test_sample_no_trigger(self) -> None:
        synth = FedChaosSynthesizer(_make_config(point="global_init"))
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "chaos_col" in result.columns

    def test_sample_crash(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="crash", point="synth_sample")
        )
        with pytest.raises(RuntimeError, match="Chaos:.*synth_sample"):
            synth.sample(Update(), num_rows=5, seed=42)

    def test_sample_corrupt(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="corrupt", point="synth_sample")
        )
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "WRONG_COL_1" in result.columns
        assert result["WRONG_COL_1"].isna().all()

    def test_sample_wrong_type(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="wrong_type", point="synth_sample")
        )
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert isinstance(result, dict)
        assert result == {"THIS_IS": "NOT_A_DATAFRAME"}

    def test_sample_empty(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="empty", point="synth_sample")
        )
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_sample_large_payload(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="large_payload", point="synth_sample", intensity=0.01)
        )
        with pytest.raises(MemoryError, match="Chaos: payload too large"):
            synth.sample(Update(), num_rows=5, seed=42)

    def test_fed_init_no_trigger(self) -> None:
        synth = FedChaosSynthesizer(_make_config(point="global_init"))
        schema = TableSchema()
        result = synth.fed_init(Update(), 42, schema, pd.DataFrame({"a": [1]}))
        assert isinstance(result, Update)
        assert result.is_empty()

    def test_fed_init_crash(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="crash", point="synth_fed_init")
        )
        with pytest.raises(RuntimeError, match="Chaos:.*synth_fed_init"):
            synth.fed_init(Update(), 42, TableSchema(), pd.DataFrame({"a": [1]}))

    def test_fed_init_corrupt(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="corrupt", point="synth_fed_init")
        )
        result = synth.fed_init(Update(), 42, TableSchema(), pd.DataFrame({"a": [1]}))
        assert isinstance(result, Update)
        assert "corrupt-nan" in result.arrays

    def test_sample_delay(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="delay", point="synth_sample", intensity=0.0)
        )
        with patch("fedbench.algorithms.fed_chaos.time.sleep") as mock_sleep:
            result = synth.sample(Update(), num_rows=5, seed=42)
            mock_sleep.assert_called_once_with(0.0)
        # delay doesn't replace the return value, so default kicks in
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_sample_leak(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="leak", point="synth_sample", intensity=0.001)
        )
        initial_count = len(_leak_store)
        result = synth.sample(Update(), num_rows=5, seed=42)
        assert len(_leak_store) == initial_count + 1
        # leak doesn't replace return value
        assert isinstance(result, pd.DataFrame)


# ── Exception types ─────────────────────────────────────────────────


class TestCrashExceptions:
    @pytest.mark.parametrize(
        "exc_name,exc_type",
        [
            ("RuntimeError", RuntimeError),
            ("ValueError", ValueError),
            ("TypeError", TypeError),
            ("OverflowError", OverflowError),
            ("NotImplementedError", NotImplementedError),
            ("OSError", OSError),
        ],
    )
    def test_crash_raises_configured_exception(
        self, exc_name: str, exc_type: type
    ) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="crash", point="synth_train", exception=exc_name)
        )
        with pytest.raises(exc_type, match="Chaos:"):
            synth.train(Update(), pd.DataFrame({"a": [1]}))

    def test_crash_memory_error(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(scenario="crash", point="synth_train", exception="MemoryError")
        )
        with pytest.raises(MemoryError, match="Chaos:"):
            synth.train(Update(), pd.DataFrame({"a": [1]}))

    def test_crash_stop_iteration(self) -> None:
        synth = FedChaosSynthesizer(
            _make_config(
                scenario="crash", point="synth_train", exception="StopIteration"
            )
        )
        with pytest.raises(StopIteration, match="Chaos:"):
            synth.train(Update(), pd.DataFrame({"a": [1]}))


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    def test_fed_chaos_registered(self) -> None:
        from fedbench.algorithms import register_builtin_algorithms
        from fedbench.core.algorithm import Algorithm
        from fedbench.runtime.registry import FactoryRegistry

        registry: FactoryRegistry[Algorithm] = FactoryRegistry(
            group="fedbench.test.algorithms",
            product_cls=Algorithm,
        )
        register_builtin_algorithms(registry)
        assert "fed_chaos" in registry
