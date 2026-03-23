from collections.abc import Iterable
from typing import cast

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


class FedHelloCoordinator(SingleStepCoordinator):
    def __init__(self, name: str) -> None:
        self._name = name
        self._state: list[NDArray[np.int_]] | None = None
        self._df: pd.DataFrame | None = None

    @property
    def global_state(self) -> Update | None:
        return self._create_update()

    def attach_global_init_artifacts(self, artifacts: Update) -> None:
        # noinspection PyUnnecessaryCast
        self._state = cast(list[NDArray[np.int_]], artifacts.arrays["state"])

    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        replies = list(replies)
        reply = replies[0][1]
        self._df = reply.objects["objects"]["df"]
        log_info(str(self), f"Hello from aggregate_train, {self._name}!")

    def _create_update(self) -> Update:
        # noinspection PyUnnecessaryCast
        state = cast(list[NDArray[np.int_]], self._state)
        update = Update(arrays={"state": state})
        if self._df is not None:
            update.objects["objects"] = {"df": self._df}
        return update


class FedHelloSynthesizer(Synthesizer):
    def __init__(self, name: str) -> None:
        self._name = name
        self._cache: Update | None = None

    def attach_client_cache(self, cache: Update) -> None:
        self._cache = cache
        if "counters" not in self._cache.metrics:
            self._cache.metrics["counters"] = {"train": 0, "sample": 0}

    def train(
        self,
        request: Update,
        data: pd.DataFrame,
    ) -> Update:

        if self._cache is not None:
            self._cache.metrics["counters"]["train"] += 1  # type: ignore[operator]
            count = self._cache.metrics["counters"]["train"]
        else:
            count = None

        log_info(str(self), f"Hello {count} from train, {self._name}!")
        return Update(arrays=request.arrays, objects={"objects": {"df": data}})

    def sample(
        self,
        request: Update,
        num_rows: int,
        seed: int,
    ) -> pd.DataFrame:

        if self._cache is not None:
            self._cache.metrics["counters"]["sample"] += 1  # type: ignore[operator]
            count = self._cache.metrics["counters"]["sample"]
        else:
            count = None

        log_info(str(self), f"Hello {count} from sample, {self._name}!")
        # noinspection PyUnnecessaryCast
        try:
            return cast(pd.DataFrame, request.objects["objects"]["df"])[:num_rows]
        except IndexError:
            return cast(pd.DataFrame, request.objects["objects"]["df"])


class FedHello(Algorithm):
    """Say a federated hello."""

    def __init__(self, name: str = "Stranger") -> None:
        self._coord_factory = lambda: FedHelloCoordinator(name)
        self._synth_factory = lambda: FedHelloSynthesizer(name)

    @property
    def coordinator_spec(self) -> ComponentSpec[Coordinator]:
        return coordinator_spec(self._coord_factory)

    @property
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        return synthesizer_spec(self._synth_factory)

    def global_init(
        self, seed: int, schema: TableSchema, dataset: DataFrame
    ) -> GlobalInitArtifacts | None:

        rng = np.random.default_rng(seed)
        return GlobalInitArtifacts(
            coordinator=Update(
                arrays={"state": [rng.integers(0, 100, (3, 3)) for _ in range(10)]},
            )
        )
