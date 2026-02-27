from collections.abc import Iterable

import pandas as pd

from fedbench.core.algorithm import Aggregator, Algorithm
from fedbench.core.algorithm import Synthesizer
from fedbench.core.data import load_csv
from fedbench.core.logger import debug_calls
from fedbench.core.update import Update

_bullshit_state = {
    "bs1": {"bs11": 1, "bs12": 2, "bs13": 3, "nested_dict": {1: 1}},
    "bs2": {"bs21": 1, "bs22": 2, "bs23": 3, "nested_dict": {2: 2}},
}


class FedNoop(Algorithm):
    @classmethod
    def create_aggregator(cls) -> Aggregator:
        return FedSmokeAggregator()

    @classmethod
    def create_synthesizer(cls) -> Synthesizer:
        return FedSmokeSynthesizer()


class FedSmokeAggregator(Aggregator):
    @debug_calls(__name__)
    def aggregate_init(self, replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-bs-state"] = _bullshit_state
        return update

    @debug_calls(__name__)
    def aggregate_train(
            self,
            replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-bs-state"] = _bullshit_state
        return update


class FedSmokeSynthesizer(Synthesizer):
    @debug_calls(__name__)
    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        update = Update()
        update.objects["my-bs-state"] = _bullshit_state
        return update

    @debug_calls(__name__)
    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int) -> pd.DataFrame:

        # TODO: Extremely hacky, fix later
        df, schema = load_csv("datasets/breast_cancer.csv")
        return df

        #num_records = 10
        #data = {
        #    "ints": np.random.randint(0, 100, size=num_records),
        #    "floats": np.random.randn(num_records),
        #    "dates": pd.date_range('2023-01-01', periods=num_records),
        #    "categories": np.random.choice(["a", "b", "c"], size=num_records)
        #}
        #return pd.DataFrame(data)