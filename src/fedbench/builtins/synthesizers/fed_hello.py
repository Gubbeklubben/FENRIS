import json
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from fedbench.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fedbench.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fedbench.core.data import TableSchema
from fedbench.core.encoder import FedbenchEncoder
from fedbench.core.logger import log_info
from fedbench.core.payload import ArraysTarget, Payload


class FedHello(Synthesizer):
    """Say a federated hello."""

    def __init__(self, name: str = "Stranger") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return "fed_hello"

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.NUMPY

    @property
    def supports_coordinators(self) -> set[str]:
        return {"fedavg"}

    def global_init(
        self, dataset: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:

        rng = np.random.default_rng(context.seed)

        return GlobalInitArtifacts(
            synthesizer=Payload(
                extras={
                    "extras": {
                        "schema": json.dumps(context.schema, cls=FedbenchEncoder)
                    }
                },
            ),
            coordinator=GlobalState(
                [rng.integers(0, 100, (3, 3)) for _ in range(10)]
            ).encode(),
        )

    def train(
        self,
        request: Payload,
        data: DataFrame,
        context: TrainContext,
    ) -> Payload:

        if context.client_cache is not None:
            try:
                context.client_cache.metrics["counters"]["train"] += 1  # type: ignore[operator]
            except KeyError:
                context.client_cache.metrics["counters"] = {"train": 1}

            count = context.client_cache.metrics["counters"]["train"]
            log_info(str(self), f"Hello {count} from train, {self._name}!")
        else:
            log_info(str(self), f"Hello from train {self._name}!")

        return ClientUpdate(state=GlobalState.decode(request).state, count=1).encode()

    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> pd.DataFrame:

        if context.client_cache is not None:
            try:
                context.client_cache.metrics["counters"]["sample"] += 1  # type: ignore[operator]
            except KeyError:
                context.client_cache.metrics["counters"] = {"sample": 1}

            count = context.client_cache.metrics["counters"]["sample"]
            log_info(str(self), f"Hello {count} from sample, {self._name}!")
        else:
            log_info(str(self), f"Hello from sample {self._name}!")

        # noinspection PyUnnecessaryCast
        artifacts = cast(Payload, context.global_init_artifacts)
        # noinspection PyUnnecessaryCast
        schema_jsons = cast(str, artifacts.extras["extras"]["schema"])
        return self._dummy_df(
            json.loads(schema_jsons, object_hook=FedbenchEncoder.decode)
        )

    @staticmethod
    def _dummy_df(schema: TableSchema) -> pd.DataFrame:
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
        return pd.DataFrame([row], columns=[c.name for c in schema.columns])
