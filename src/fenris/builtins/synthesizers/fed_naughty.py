r"""Naughty synthesizer for testing the framework.

Intentionally injects errors, corrupts data, and violates protocol
assumptions at configurable lifecycle points. Used to discover weak spots.

Usage:
    fenris run fed_naughty iid_partitioner datasets/heart_disease.csv \\
        --algorithm-kwargs "scenario=crash,point=synth_train,exception=MemoryError"
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from fenris.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fenris.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fenris.core.logger import log_info
from fenris.core.payload import ArraysTarget, Payload

_LOG_SRC = __name__

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

_VALID_SCENARIOS = frozenset({
    "crash",
    "corrupt",
    "nan_columns",
    "wrong_type",
    "empty",
})

_VALID_POINTS = frozenset({
    "global_init",
    "synth_train",
    "synth_sample",
})

_INVALID_COMBINATIONS = frozenset({
    ("nan_columns", "global_init"),
    ("nan_columns", "synth_train"),
})


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
        if (self.scenario, self.point) in _INVALID_COMBINATIONS:
            raise ValueError(
                f"Scenario '{self.scenario}' is not applicable at point '{self.point}'."
            )


def _do_crash(config: _NaughtyConfig, point: str) -> None:
    exc_cls = _EXCEPTION_MAP[config.exception]
    raise exc_cls(f"Naughty: {config.exception} at {point}")


def _do_corrupt_payload(point: str) -> Payload:
    """Return a Payload filled with NaN/inf values."""
    log_info(_LOG_SRC, f"Returning corrupt Payload at {point}")
    nan_arr: NDArray[Any] = np.full((4, 4), np.nan)
    inf_arr: NDArray[Any] = np.full((4, 4), np.inf)
    return Payload(
        arrays={"corrupt-nan": [nan_arr], "corrupt-inf": [inf_arr]},
        metrics={"corrupt": {"nan": math.nan, "inf": math.inf}},
    )


def _encode_artifacts(context: GlobalInitContext, config: _NaughtyConfig) -> Payload:
    return Payload(
        extras={
            "naughty": {
                "scenario": config.scenario,
                "point": config.point,
                "exception": config.exception,
            },
            "schema": {
                "column_names": [col.name for col in context.schema.columns],
            },
        }
    )


def _decode_config(artifacts: Payload | None) -> _NaughtyConfig | None:
    if artifacts is None:
        return None
    try:
        naughty = dict(artifacts.extras["naughty"])
        # noinspection PyUnnecessaryCast
        return _NaughtyConfig(
            scenario=cast(str, naughty["scenario"]),
            point=cast(str, naughty["point"]),
            exception=cast(str, naughty["exception"]),
        )
    except KeyError:
        return None


def _decode_schema_columns(artifacts: Payload | None) -> list[str] | None:
    if artifacts is None:
        return None
    try:
        # noinspection PyUnnecessaryCast
        return cast(list[str], artifacts.extras["schema"]["column_names"])
    except KeyError:
        return None


class FedNaughty(Synthesizer):
    """CLI-configurable naughty synthesizer for framework testing."""

    SUPPORTED_COORDINATORS = {"fedavg"}

    def __init__(
        self,
        scenario: str = "nan_columns",
        point: str = "synth_sample",
        exception: str = "",
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

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.TORCH

    def global_init(
        self,
        df: DataFrame,
        context: GlobalInitContext,
    ) -> GlobalInitArtifacts:
        if self._config.point == "global_init":
            scenario = self._config.scenario
            log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at global_init")

            if scenario == "crash":
                _do_crash(self._config, "global_init")
            elif scenario == "corrupt":
                corrupt = _do_corrupt_payload("global_init")
                return GlobalInitArtifacts(coordinator=corrupt, synthesizer=corrupt)
            elif scenario == "wrong_type":
                log_info(_LOG_SRC, "Returning wrong type from global_init")
                return "NOT_ARTIFACTS"  # type: ignore[return-value]
            elif scenario == "empty":
                return GlobalInitArtifacts(coordinator=Payload(), synthesizer=Payload())

        return GlobalInitArtifacts(
            coordinator=GlobalState(state={}).encode(),
            synthesizer=_encode_artifacts(context, self._config),
        )

    def train(
        self,
        request: Payload,
        df: DataFrame,
        context: TrainContext,
    ) -> Payload:
        config = _decode_config(context.global_init_artifacts) or self._config

        if config.point == "synth_train":
            scenario = config.scenario
            log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at synth_train")

            if scenario == "crash":
                _do_crash(config, "synth_train")
            elif scenario == "corrupt":
                return _do_corrupt_payload("synth_train")
            elif scenario == "wrong_type":
                return "THIS_IS_NOT_A_PAYLOAD"  # type: ignore[return-value]
            elif scenario == "empty":
                return Payload()

        return ClientUpdate(
            state=GlobalState.decode(request).state, count=len(df)
        ).encode()

    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> DataFrame:
        config = _decode_config(context.global_init_artifacts) or self._config
        column_names = _decode_schema_columns(context.global_init_artifacts) or []

        if config.point == "synth_sample":
            scenario = config.scenario
            log_info(_LOG_SRC, f"NAUGHTY [{scenario}] triggered at synth_sample")

            if scenario == "crash":
                _do_crash(config, "synth_sample")
            elif scenario == "corrupt":
                return pd.DataFrame({
                    "WRONG_COL_1": [math.nan] * context.num_rows,
                    "WRONG_COL_2": [math.inf] * context.num_rows,
                })
            elif scenario == "nan_columns":
                return pd.DataFrame({
                    name: [math.nan] * context.num_rows for name in column_names
                })
            elif scenario == "wrong_type":
                return {"THIS_IS": "NOT_A_DATAFRAME"}  # type: ignore[return-value]
            elif scenario == "empty":
                return pd.DataFrame()

        rng = np.random.default_rng(context.seed)
        return pd.DataFrame({
            name: rng.random(context.num_rows) for name in column_names
        })
