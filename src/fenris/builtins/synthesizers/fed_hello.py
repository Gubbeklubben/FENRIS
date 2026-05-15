import json
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from fenris.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fenris.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fenris.core.data import TableSchema
from fenris.core.encoder import FenrisEncoder
from fenris.core.logger import log_info
from fenris.core.payload import ArraysTarget, Payload


class FedHello(Synthesizer):
    """Say a federated hello."""

    SUPPORTED_COORDINATORS = {"fedavg"}

    def __init__(self, name: str = "Stranger") -> None:
        self._name = name

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    def global_init(
        self, df: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:

        rng = np.random.default_rng(context.seed)

        return GlobalInitArtifacts(
            synthesizer=Payload(
                extras={
                    "extras": {"schema": json.dumps(context.schema, cls=FenrisEncoder)}
                },
            ),
            coordinator=GlobalState(
                [rng.integers(0, 100, (3, 3)) for _ in range(10)]
            ).encode(),
        )

    def train(
        self,
        request: Payload,
        df: DataFrame,
        context: TrainContext,
    ) -> Payload:

        if context.client_storage is not None:
            try:
                context.client_storage.metrics["counters"]["train"] += 1  # type: ignore[operator]
            except KeyError:
                context.client_storage.metrics["counters"] = {"train": 1}

            count = context.client_storage.metrics["counters"]["train"]
            log_info(str(self), f"Hello {count} from train, {self._name}!")
        else:
            log_info(str(self), f"Hello from train {self._name}!")

        return ClientUpdate(state=GlobalState.decode(request).state, count=1).encode()

    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> pd.DataFrame:

        if context.client_storage is not None:
            try:
                context.client_storage.metrics["counters"]["sample"] += 1  # type: ignore[operator]
            except KeyError:
                context.client_storage.metrics["counters"] = {"sample": 1}

            count = context.client_storage.metrics["counters"]["sample"]
            log_info(str(self), f"Hello {count} from sample, {self._name}!")
        else:
            log_info(str(self), f"Hello from sample {self._name}!")

        # noinspection PyUnnecessaryCast
        artifacts = cast(Payload, context.global_init_artifacts)
        # noinspection PyUnnecessaryCast
        schema_jsons = cast(str, artifacts.extras["extras"]["schema"])
        return self._dummy_df(
            json.loads(schema_jsons, object_hook=FenrisEncoder.decode),
            context.num_rows,
        )

    @staticmethod
    def _dummy_df(schema: TableSchema, num_rows: int) -> pd.DataFrame:
        defaults = {
            "continuous": 0.0,
            "integer": 0,
            "categorical": "unknown",
            "binary": "no",
        }
        row = {
            c.name: defaults.get(getattr(c.kind, "value", c.kind), None)
            for c in schema.columns
        }
        return pd.DataFrame([row] * num_rows, columns=[c.name for c in schema.columns])
