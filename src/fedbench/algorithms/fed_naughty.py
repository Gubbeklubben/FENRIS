"""Naughty algorithm for testing the framework.

Intentionally injects errors, corrupts data, and violates protocol
assumptions at configurable lifecycle points. Used to discover weak spots.

Usage:
    fedbench run fed_naughty iid_partitioner datasets/heart_disease.csv \\
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

_LOG_SRC = __name__

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
    "Exception": Exception,
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
        "coord_train",
        "synth_train",
        "synth_sample",
    }
)


@dataclass(frozen=True)
class _NaughtyConfig:
    scenario: str
    point: str
    exception: str

    def __post_init__(self) -> None:
        if self.scenario not in _VALID_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. Valid: {sorted(_VALID_SCENARIOS)}"
            )
        if self.point not in _VALID_POINTS:
            raise ValueError(
                f"Unknown point '{self.point}'. Valid: {sorted(_VALID_POINTS)}"
            )
        if self.scenario == "crash" and self.exception not in _EXCEPTION_MAP:
            raise ValueError(
                f"Unknown exception '{self.exception}'. Valid: {sorted(_EXCEPTION_MAP)}"
            )


# ── Scenario helpers ─────────────────────────────────────────────────


def _do_crash(config: _NaughtyConfig, point: str) -> None:
    exc_cls = _EXCEPTION_MAP[config.exception]
    raise exc_cls(f"Naughty: {config.exception} at {point}")


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


class FedNaughtyCoordinator(SingleStepCoordinator):
    def __init__(self, config: _NaughtyConfig) -> None:
        self._config = config
        self._state: Update = Update()

    @property
    def global_state(self) -> Update:
        return self._state

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
        """Dispatch to the configured scenario."""
        scenario = self._config.scenario
        log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at {point}")

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


class FedNaughtySynthesizer(Synthesizer):
    def __init__(self, config: _NaughtyConfig) -> None:
        self._config = config

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
        return pd.DataFrame({"naughty_col": rng.random(num_rows)})

    def _trigger(self, point: str) -> Any:
        """Dispatch scenarios that return Update-like objects."""
        scenario = self._config.scenario
        log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at {point}")

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
        log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at {point}")

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


class FedNaughty(Algorithm):
    """CLI-configurable naughty algorithm for framework testing."""

    def __init__(
        self,
        scenario: str = "crash",
        point: str = "synth_train",
        exception: str = "Exception",
    ) -> None:

        self._config = _NaughtyConfig(
            scenario=scenario,
            point=point,
            exception=exception,
        )
        log_info(
            _LOG_SRC,
            f"Configured: scenario={scenario}, point={point}, exception={exception}",
        )
        self._coord_factory = lambda: FedNaughtyCoordinator(self._config)
        self._synth_factory = lambda: FedNaughtySynthesizer(self._config)

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
        log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at global_init")

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
