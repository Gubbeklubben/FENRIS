"""Chaos testing algorithm for stress-testing the FedBench framework.

Intentionally injects errors, corrupts data, and violates protocol
assumptions at configurable lifecycle points. Used to discover weak spots
in error handling, validation, and resource management.

Usage:
    fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \\
        --algorithm-kwargs "scenario=crash,point=synth_train,exception=MemoryError"
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generator

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
from fedbench.core.logger import log_error, log_info, log_warning
from fedbench.core.update import Update

_LOG_SRC = "fed_chaos"

# Exception classes that can be raised via the `exception` parameter.
_EXCEPTION_MAP: dict[str, type[BaseException]] = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "MemoryError": MemoryError,
    "KeyboardInterrupt": KeyboardInterrupt,
    "SystemExit": SystemExit,
    "StopIteration": StopIteration,
    "OverflowError": OverflowError,
    "ZeroDivisionError": ZeroDivisionError,
    "NotImplementedError": NotImplementedError,
    "OSError": OSError,
}

_VALID_SCENARIOS = frozenset(
    {
        "crash",
        "delay",
        "leak",
        "corrupt",
        "wrong_type",
        "empty",
        "infinite_loop",
        "large_payload",
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
    intensity: float
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

# Persistent leak storage — intentionally never freed.
_leak_store: list[bytearray] = []


def _do_crash(config: _ChaosConfig, point: str) -> None:
    exc_cls = _EXCEPTION_MAP[config.exception]
    raise exc_cls(f"Chaos: {config.exception} at {point}")


def _do_delay(config: _ChaosConfig, point: str) -> None:
    seconds = config.intensity
    log_warning(_LOG_SRC, f"Injecting {seconds}s delay at {point}")
    time.sleep(seconds)


def _do_leak(config: _ChaosConfig, point: str) -> None:
    megabytes = config.intensity
    nbytes = int(megabytes * 1024 * 1024)
    log_warning(_LOG_SRC, f"Leaking {megabytes} MB at {point}")
    _leak_store.append(bytearray(nbytes))


def _do_corrupt_update(point: str) -> Update:
    """Return an Update filled with NaN/inf values."""
    log_warning(_LOG_SRC, f"Returning corrupt Update at {point}")
    nan_arr: NDArray[Any] = np.full((4, 4), np.nan)
    inf_arr: NDArray[Any] = np.full((4, 4), np.inf)
    return Update(
        arrays={"corrupt-nan": [nan_arr], "corrupt-inf": [inf_arr]},
        metrics={"corrupt": {"nan": float("nan"), "inf": float("inf")}},
    )


def _do_corrupt_dataframe(
    point: str,
    num_rows: int,
) -> DataFrame:
    """Return a DataFrame with wrong schema and NaN values."""
    log_warning(_LOG_SRC, f"Returning corrupt DataFrame at {point}")
    return pd.DataFrame(
        {
            "WRONG_COL_1": [float("nan")] * num_rows,
            "WRONG_COL_2": [float("inf")] * num_rows,
            "WRONG_COL_3": ["not_a_number"] * num_rows,
        }
    )


def _do_wrong_type_update(point: str) -> Any:
    log_warning(_LOG_SRC, f"Returning wrong type (str) instead of Update at {point}")
    return "THIS_IS_NOT_AN_UPDATE"


def _do_wrong_type_dataframe(point: str) -> Any:
    log_warning(
        _LOG_SRC, f"Returning wrong type (dict) instead of DataFrame at {point}"
    )
    return {"THIS_IS": "NOT_A_DATAFRAME"}


def _do_large_payload(config: _ChaosConfig, point: str) -> None:
    megabytes = config.intensity
    nbytes = int(megabytes * 1024 * 1024)
    log_warning(_LOG_SRC, f"Allocating {megabytes} MB at {point}")
    buf: NDArray[Any] = np.zeros(nbytes // 8, dtype=np.float64)  # 8 bytes per float64
    buf[0]  # ensure the array is materialised before raising
    raise MemoryError(f"Chaos: payload too large ({megabytes} MB) at {point}")


# ── Coordinator ──────────────────────────────────────────────────────


class FedChaosCoordinator(SingleStepCoordinator):
    """Server-side chaos injector."""

    def __init__(self, config: _ChaosConfig) -> None:
        self._config = config
        self._state: Update = Update()
        self._call_count = 0

    @property
    def global_state(self) -> Update:
        return self._state

    def configure_fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Iterable[tuple[int, Update]]:
        log_info(_LOG_SRC, "FedChaosCoordinator.configure_fed_init")

        if self._config.point == "coord_fed_init":
            self._trigger("coord_fed_init")

        # Minimal passthrough — send empty update to every client.
        for cid in client_ids:
            yield cid, Update()

    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        log_info(_LOG_SRC, "FedChaosCoordinator.aggregate_fed_init")
        # Consume replies.
        for _ in replies:
            pass

    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        log_info(_LOG_SRC, "FedChaosCoordinator.aggregate_train")
        self._call_count += 1

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

    def train(
        self,
        client_ids: Iterable[int],
    ) -> Generator[
        Iterable[tuple[int, Update]],
        Iterable[tuple[int, Update]],
        None,
    ]:
        """Override train() directly for generator-level chaos."""
        if (
            self._config.scenario == "infinite_loop"
            and self._config.point == "coord_train"
        ):
            log_warning(_LOG_SRC, "Entering infinite generator loop at coord_train")
            round_num = 0
            cids = list(client_ids)
            while True:
                round_num += 1
                log_info(_LOG_SRC, f"Infinite loop: internal round {round_num}")
                updates = [(cid, Update()) for cid in cids]
                replies = yield updates
                # Consume replies and loop forever.
                for _cid, reply in replies:
                    self._state = reply

        # For all other scenarios, delegate to SingleStepCoordinator.
        yield from super().train(client_ids)

    def _trigger(self, point: str) -> Any:
        """Dispatch to the configured chaos scenario."""
        scenario = self._config.scenario
        log_error(_LOG_SRC, f"CHAOS [{scenario}] triggered at {point}")

        if scenario == "crash":
            _do_crash(self._config, point)
        elif scenario == "delay":
            _do_delay(self._config, point)
        elif scenario == "leak":
            _do_leak(self._config, point)
        elif scenario == "corrupt":
            return _do_corrupt_update(point)
        elif scenario == "wrong_type":
            return None
        elif scenario == "empty":
            self._state = Update()
            return Update()
        elif scenario == "large_payload":
            _do_large_payload(self._config, point)
        return None


# ── Synthesizer ──────────────────────────────────────────────────────


class FedChaosSynthesizer(Synthesizer):
    """Client-side chaos injector."""

    def __init__(self, config: _ChaosConfig) -> None:
        self._config = config
        self._call_count = 0

    def fed_init(
        self,
        request: Update,
        seed: int,
        schema: TableSchema,
        data: DataFrame,
    ) -> Update:
        log_info(_LOG_SRC, "FedChaosSynthesizer.fed_init")

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
        log_info(_LOG_SRC, "FedChaosSynthesizer.train")
        self._call_count += 1

        if self._config.point == "synth_train":
            result = self._trigger("synth_train")
            if result is not None:
                return result  # type: ignore[no-any-return]

        # Default passthrough: echo request back with a shape metric.
        return Update(
            metrics={"chaos": {"rows": len(data), "round": self._call_count}},
        )

    def sample(
        self,
        request: Update,
        num_rows: int,
        seed: int,
    ) -> DataFrame:
        log_info(_LOG_SRC, f"FedChaosSynthesizer.sample (n={num_rows})")

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
        log_error(_LOG_SRC, f"CHAOS [{scenario}] triggered at {point}")

        if scenario == "crash":
            _do_crash(self._config, point)
        elif scenario == "delay":
            _do_delay(self._config, point)
        elif scenario == "leak":
            _do_leak(self._config, point)
        elif scenario == "corrupt":
            return _do_corrupt_update(point)
        elif scenario == "wrong_type":
            return _do_wrong_type_update(point)
        elif scenario == "empty":
            return Update()
        elif scenario == "large_payload":
            _do_large_payload(self._config, point)
        return None

    def _trigger_sample(self, point: str, num_rows: int, seed: int) -> Any:
        """Dispatch scenarios that return DataFrame-like objects."""
        scenario = self._config.scenario
        log_error(_LOG_SRC, f"CHAOS [{scenario}] triggered at {point}")

        if scenario == "crash":
            _do_crash(self._config, point)
        elif scenario == "delay":
            _do_delay(self._config, point)
        elif scenario == "leak":
            _do_leak(self._config, point)
        elif scenario == "corrupt":
            return _do_corrupt_dataframe(point, num_rows)
        elif scenario == "wrong_type":
            return _do_wrong_type_dataframe(point)
        elif scenario == "empty":
            return pd.DataFrame()
        elif scenario == "large_payload":
            _do_large_payload(self._config, point)
        return None


# ── Algorithm ────────────────────────────────────────────────────────


class FedChaos(Algorithm):
    """CLI-configurable chaos injection algorithm for framework stress testing."""

    def __init__(
        self,
        scenario: str = "crash",
        point: str = "synth_train",
        intensity: float = 1.0,
        exception: str = "RuntimeError",
    ) -> None:
        self._config = _ChaosConfig(
            scenario=scenario,
            point=point,
            intensity=float(intensity),
            exception=exception,
        )
        log_info(
            _LOG_SRC,
            f"Configured: scenario={scenario}, point={point}, "
            f"intensity={intensity}, exception={exception}",
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
        log_info(
            _LOG_SRC, f"global_init (rows={len(dataset)}, cols={len(schema.columns)})"
        )

        if self._config.point == "global_init":
            return self._trigger_global_init()

        return None

    def _trigger_global_init(self) -> GlobalInitArtifacts | None:
        scenario = self._config.scenario
        log_error(_LOG_SRC, f"CHAOS [{scenario}] triggered at global_init")

        if scenario == "crash":
            _do_crash(self._config, "global_init")
        elif scenario == "delay":
            _do_delay(self._config, "global_init")
        elif scenario == "leak":
            _do_leak(self._config, "global_init")
        elif scenario == "corrupt":
            corrupt = _do_corrupt_update("global_init")
            return GlobalInitArtifacts(coordinator=corrupt, synthesizer=corrupt)
        elif scenario == "wrong_type":
            log_warning(_LOG_SRC, "Returning wrong type from global_init")
            return "NOT_ARTIFACTS"  # type: ignore[return-value]
        elif scenario == "empty":
            return GlobalInitArtifacts(coordinator=Update(), synthesizer=Update())
        elif scenario == "large_payload":
            _do_large_payload(self._config, "global_init")

        return None
