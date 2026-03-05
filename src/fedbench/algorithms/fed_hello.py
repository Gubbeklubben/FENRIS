from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fedbench.core.algorithm import Coordinator, Algorithm, \
    SingleStepCoordinator
from fedbench.core.algorithm import Synthesizer
from fedbench.core.data import TableSchema
from fedbench.core.logger import log_info
from fedbench.core.update import Update


class FedHello(Algorithm):
    def __init__(self, name: str = "Stranger") -> None:
        self._name = name

    def create_coordinator(self) -> Coordinator:
        return FedHelloCoordinator(self._name)

    def create_synthesizer(self) -> Synthesizer:
        return FedHelloSynthesizer(self._name)


class FedHelloCoordinator(SingleStepCoordinator):
    def __init__(self, name: str) -> None:
        self._name = name
        self._state: list[NDArray[np.int_]] | None = None
        self._df: pd.DataFrame | None = None

    @property
    def global_state(self) -> Update | None:
        return self._create_update()

    def configure_fed_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int]) -> Iterable[tuple[int, Update]]:

        log_info(str(self), f"Hello from configure_fed_init, {self._name}!")
        rng = np.random.default_rng(seed)
        self._state = [rng.integers(0, 100, (3, 3)) for _ in range(10)]

        update = self._create_update()
        for cid in client_ids:
            yield cid, update

    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        log_info(str(self), f"Hello from aggregate_fed_init, {self._name}!")

    def aggregate_train(
            self,
            replies: Iterable[tuple[int, Update]]) -> None:

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

    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        log_info(str(self), f"Hello from train, {self._name}!")
        return Update(
            arrays=request.arrays,
            objects={"objects": {"df": data}}
        )

    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int) -> pd.DataFrame:

        log_info(str(self), f"Hello from sample, {self._name}!")
        # noinspection PyUnnecessaryCast
        try:
            return cast(pd.DataFrame, request.objects["objects"]["df"])[:num_rows]
        except IndexError:
            return cast(pd.DataFrame, request.objects["objects"]["df"])

