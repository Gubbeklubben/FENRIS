from collections.abc import Iterable

import numpy as np
import pandas as pd

from fedbench.algorithms.algorithm import Aggregator, Algorithm
from fedbench.algorithms.algorithm import Synthesizer
from fedbench.common import Update
from fedbench.common import log_calls


_bullshit_state = {
    "bs1": {"bs11": 1, "bs12": 2, "bs13": 3, "nested_dict": {1: 1}},
    "bs2": {"bs21": 1, "bs22": 2, "bs23": 3, "nested_dict": {2: 2}},
}


class FedNoop(Algorithm):
    @classmethod
    def requires_non_array_protocol(cls) -> str | None:
        return "pickle"

    @classmethod
    def create_aggregator(cls) -> Aggregator:
        return FedSmokeAggregator()

    @classmethod
    def create_synthesizer(cls) -> Synthesizer:
        return FedSmokeSynthesizer()


class FedSmokeAggregator(Aggregator):
    @log_calls(__name__)
    def aggregate_init(self, replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-bs-state"] = _bullshit_state
        return update

    @log_calls(__name__)
    def aggregate_train(
            self,
            replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-bs-state"] = _bullshit_state
        return update


class FedSmokeSynthesizer(Synthesizer):
    @log_calls(__name__)
    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        update = Update()
        update.objects["my-bs-state"] = _bullshit_state
        return update

    @log_calls(__name__)
    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int) -> pd.DataFrame:

        num_records = 10
        data = {
            "ints": np.random.randint(0, 100, size=num_records),
            "floats": np.random.randn(num_records),
            "dates": pd.date_range('2023-01-01', periods=num_records),
            "categories": np.random.choice(["a", "b", "c"], size=num_records)
        }
        return pd.DataFrame(data)