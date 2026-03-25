import json
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from fedbench.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fedbench.core.algorithm import (
    Algorithm,
    ComponentSpec,
    GlobalInitArtifacts,
    Synthesizer,
    synthesizer_spec,
)
from fedbench.core.data import TableSchema
from fedbench.core.encoder import FedbenchEncoder
from fedbench.core.logger import log_info
from fedbench.core.payload import Payload


class FedHelloSynthesizer(Synthesizer):
    def __init__(self, name: str) -> None:
        self._name = name
        self._schema: TableSchema | None = None
        self._cache: Payload | None = None

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        jsons = artifacts.extras["extras"]["schema"]
        # noinspection PyUnnecessaryCast
        self._schema = json.loads(cast(str, jsons), object_hook=FedbenchEncoder.decode)

    def attach_client_cache(self, cache: Payload) -> None:
        self._cache = cache
        if "counters" not in self._cache.metrics:
            self._cache.metrics["counters"] = {"train": 0, "sample": 0}

    def train(
        self,
        request: Payload,
        data: pd.DataFrame,
    ) -> Payload:

        if self._cache is not None:
            self._cache.metrics["counters"]["train"] += 1  # type: ignore[operator]
            count = self._cache.metrics["counters"]["train"]
        else:
            count = None

        if count is not None:
            log_info(str(self), f"Hello {count} from train, {self._name}!")

        return ClientUpdate(state=GlobalState.decode(request).state, count=1).encode()

    def sample(
        self,
        request: Payload,
        num_rows: int,
        seed: int,
    ) -> pd.DataFrame:

        if self._cache is not None:
            self._cache.metrics["counters"]["sample"] += 1  # type: ignore[operator]
            count = self._cache.metrics["counters"]["sample"]
        else:
            count = None

        if count is not None:
            log_info(str(self), f"Hello {count} from sample, {self._name}!")

        return self._dummy_df()

    def _dummy_df(self) -> pd.DataFrame:
        if self._schema is None:
            return pd.DataFrame()

        defaults = {
            "continuous": 0.0,
            "integer": 0,
            "categorical": "unknown",
            "binary": "no",
        }
        row = {
            c.name: defaults.get(getattr(c.kind, "value", c.kind), None)
            for c in self._schema.columns
        }
        return pd.DataFrame([row], columns=[c.name for c in self._schema.columns])


class FedHello(Algorithm):
    """Say a federated hello."""

    def __init__(self, name: str = "Stranger") -> None:
        self._synth_factory = lambda: FedHelloSynthesizer(name)

    @property
    def supports_coordinators(self) -> set[str]:
        return set("fedavg")

    @property
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        return synthesizer_spec(self._synth_factory)

    def global_init(
        self, seed: int, schema: TableSchema, dataset: DataFrame
    ) -> GlobalInitArtifacts | None:

        rng = np.random.default_rng(seed)

        return GlobalInitArtifacts(
            synthesizer=Payload(
                extras={"extras": {"schema": json.dumps(schema, cls=FedbenchEncoder)}},
            ),
            coordinator=GlobalState(
                [rng.integers(0, 100, (3, 3)) for _ in range(10)]
            ).encode(),
        )
