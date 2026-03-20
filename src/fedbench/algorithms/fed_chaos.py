"""Chaos testing algorithm for stress-testing the FedBench framework.

Intentionally injects errors, corrupts data, and violates protocol
assumptions at configurable lifecycle points. Used to discover weak spots
in error handling, validation, and resource management.

Usage:
    fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \\
        --algorithm-kwargs "scenario=crash,point=synth_train,exception=MemoryError"
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from fedbench.core.algorithm import (
    Algorithm,
    ComponentSpec,
    Coordinator,
    GlobalInitArtifacts,
    SingleStepCoordinator,
    Synthesizer,
    coordinator_spec,
    synthesizer_spec,
)
from fedbench.core.data import TableSchema
from fedbench.core.logger import log_info
from fedbench.core.update import Update

_LOG_SRC = "fed_chaos"

# Exception classes that can be raised via the `exception` parameter.
_EXCEPTION_MAP: dict[str, type[Exception]] = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "MemoryError": MemoryError,
    "OverflowError": OverflowError,
    "ZeroDivisionError": ZeroDivisionError,
    "NotImplementedError": NotImplementedError,
    "OSError": OSError,
}

_VALID_SCENARIOS = frozenset(
    {
        "crash",
        "corrupt",
        "wrong_type",
        "empty",
    }
)

_VALID_POINTS = frozenset(
    {
        "global_init",
        "coord_fed_init",
        "coord_train",
        "synth_fed_init",
        "synth_train",
        "synth_sample",
    }
)


@dataclass(frozen=True)
class _ChaosConfig:
    """Immutable chaos injection configuration."""

    scenario: str
    point: str
    exception: str

    def __post_init__(self) -> None:
        if self.scenario not in _VALID_SCENARIOS:
            raise ValueError(
                f"Unknown chaos scenario '{self.scenario}'. "
                f"Valid: {sorted(_VALID_SCENARIOS)}"
            )
        if self.point not in _VALID_POINTS:
            raise ValueError(
                f"Unknown chaos point '{self.point}'. Valid: {sorted(_VALID_POINTS)}"
            )
        if self.scenario == "crash" and self.exception not in _EXCEPTION_MAP:
            raise ValueError(
                f"Unknown exception '{self.exception}'. Valid: {sorted(_EXCEPTION_MAP)}"
            )


# ── Scenario helpers ─────────────────────────────────────────────────


def _do_crash(config: _ChaosConfig, point: str) -> None:
    exc_cls = _EXCEPTION_MAP[config.exception]
    raise exc_cls(f"Chaos: {config.exception} at {point}")


def _do_corrupt_update(point: str) -> Update:
    """Return an Update filled with NaN/inf values."""
    log_info(_LOG_SRC, f"Returning corrupt Update at {point}")
    nan_arr: NDArray[Any] = np.full((4, 4), np.nan)
    inf_arr: NDArray[Any] = np.full((4, 4), np.inf)
    return Update(
        arrays={"corrupt-nan": [nan_arr], "corrupt-inf": [inf_arr]},
        metrics={"corrupt": {"nan": float("nan"), "inf": float("inf")}},
    )


# ── Coordinator ──────────────────────────────────────────────────────


class FedChaosCoordinator(SingleStepCoordinator):
    """Server-side chaos injector."""

    def __init__(self, config: _ChaosConfig) -> None:
        self._config = config
        self._state: Update = Update()

    @property
    def global_state(self) -> Update:
        return self._state

    def configure_fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Iterable[tuple[int, Update]]:

        if self._config.point == "coord_fed_init":
            self._trigger("coord_fed_init")

        # Minimal passthrough — send empty update to every client.
        for cid in client_ids:
            yield cid, Update()

    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        # Consume replies.
        for _ in replies:
            pass

    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        if self._config.point == "coord_train":
            result = self._trigger("coord_train")
            if isinstance(result, Update):
                self._state = result
                return
            # wrong_type leaves _state unchanged (keeps previous or empty initial).
            if self._config.scenario == "wrong_type":
                return

        # Default: store the first reply as global state.
        for _cid, reply in replies:
            self._state = reply
            break

    def _trigger(self, point: str) -> Any:
        """Dispatch to the configured chaos scenario."""
        scenario = self._config.scenario
        log_info(_LOG_SRC, f"CHAOS [{scenario}] triggered at {point}")

        if scenario == "crash":
            _do_crash(self._config, point)
        elif scenario == "corrupt":
            return _do_corrupt_update(point)
        elif scenario == "wrong_type":
            return None
        elif scenario == "empty":
            self._state = Update()
            return Update()
        return None


# ── Synthesizer ──────────────────────────────────────────────────────


class FedChaosSynthesizer(Synthesizer):
    """Client-side chaos injector."""

    def __init__(self, config: _ChaosConfig) -> None:
        self._config = config

    def fed_init(
        self,
        request: Update,
        seed: int,
        schema: TableSchema,
        data: DataFrame,
    ) -> Update:

        if self._config.point == "synth_fed_init":
            result = self._trigger("synth_fed_init")
            if result is not None:
                return result  # type: ignore[no-any-return]

        return Update()

    def train(
        self,
        request: Update,
        data: DataFrame,
    ) -> Update:

        if self._config.point == "synth_train":
            result = self._trigger("synth_train")
            if result is not None:
                return result  # type: ignore[no-any-return]

        return Update()

    def sample(
        self,
        request: Update,
        num_rows: int,
        seed: int,
    ) -> DataFrame:

        if self._config.point == "synth_sample":
            result = self._trigger_sample("synth_sample", num_rows, seed)
            if result is not None:
                return result  # type: ignore[no-any-return]

        # Default: return a minimal valid DataFrame of random noise.
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"chaos_col": rng.random(num_rows)})

    def _trigger(self, point: str) -> Any:
        """Dispatch scenarios that return Update-like objects."""
        scenario = self._config.scenario
        log_info(_LOG_SRC, f"CHAOS [{scenario}] triggered at {point}")

        if scenario == "crash":
            _do_crash(self._config, point)
        elif scenario == "corrupt":
            return _do_corrupt_update(point)
        elif scenario == "wrong_type":
            return "THIS_IS_NOT_AN_UPDATE"
        elif scenario == "empty":
            return Update()
        return None

    def _trigger_sample(self, point: str, num_rows: int, seed: int) -> Any:
        """Dispatch scenarios that return DataFrame-like objects."""
        scenario = self._config.scenario
        log_info(_LOG_SRC, f"CHAOS [{scenario}] triggered at {point}")

        if scenario == "crash":
            _do_crash(self._config, point)
        elif scenario == "corrupt":
            return pd.DataFrame(
                {
                    "WRONG_COL_1": [float("nan")] * num_rows,
                    "WRONG_COL_2": [float("inf")] * num_rows,
                }
            )
        elif scenario == "wrong_type":
            return {"THIS_IS": "NOT_A_DATAFRAME"}
        elif scenario == "empty":
            return pd.DataFrame()
        return None


# ── Algorithm ────────────────────────────────────────────────────────


class FedChaos(Algorithm):
    """CLI-configurable chaos injection algorithm for framework stress testing."""

    def __init__(
        self,
        scenario: str = "crash",
        point: str = "synth_train",
        exception: str = "RuntimeError",
    ) -> None:
        self._config = _ChaosConfig(
            scenario=scenario,
            point=point,
            exception=exception,
        )
        log_info(
            _LOG_SRC,
            f"Configured: scenario={scenario}, point={point}, exception={exception}",
        )
        self._coord_factory = lambda: FedChaosCoordinator(self._config)
        self._synth_factory = lambda: FedChaosSynthesizer(self._config)

    @property
    def coordinator_spec(self) -> ComponentSpec[Coordinator]:
        return coordinator_spec(self._coord_factory)

    @property
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        return synthesizer_spec(self._synth_factory)

    def global_init(
        self,
        seed: int,
        schema: TableSchema,
        dataset: DataFrame,
    ) -> GlobalInitArtifacts | None:

        if self._config.point == "global_init":
            return self._trigger_global_init()

        return None

    def _trigger_global_init(self) -> GlobalInitArtifacts | None:
        scenario = self._config.scenario
        log_info(_LOG_SRC, f"CHAOS [{scenario}] triggered at global_init")

        if scenario == "crash":
            _do_crash(self._config, "global_init")
        elif scenario == "corrupt":
            corrupt = _do_corrupt_update("global_init")
            return GlobalInitArtifacts(coordinator=corrupt, synthesizer=corrupt)
        elif scenario == "wrong_type":
            log_info(_LOG_SRC, "Returning wrong type from global_init")
            return "NOT_ARTIFACTS"  # type: ignore[return-value]
        elif scenario == "empty":
            return GlobalInitArtifacts(coordinator=Update(), synthesizer=Update())

        return None
