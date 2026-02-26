from collections.abc import Iterable

import pandas as pd
import numpy as np

from fedbench.core.algorithm import Aggregator, Synthesizer, Algorithm
from fedbench.core.data import load_csv
from fedbench.core.update import Update
from fedbench.core.logging import log_calls


_some_state = {
    "s1": {"s11": 1, "s12": 2, "s13": 3, "nested_dict": {1: 1}},
    "s2": {"s21": 1, "s22": 2, "s23": 3, "nested_dict": {2: 2}},
}


class FedTGAN(Algorithm):
    @classmethod
    def create_aggregator(cls) -> Aggregator:
        return FedTGANAggregator()

    @classmethod
    def create_synthesizer(cls) -> Synthesizer:
        return FedTGANSynthesizer()


class FedTGANAggregator(Aggregator):
    @log_calls(__name__)
    def aggregate_init(self, replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-state"] = _some_state
        return update

    @log_calls(__name__)
    def aggregate_train(
            self,
            replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-state"] = _some_state
        return update


class FedTGANSynthesizer(Synthesizer):
    @log_calls(__name__)
    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        update = Update()
        update.objects["my-state"] = _some_state
        return update

    @log_calls(__name__)
    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int) -> pd.DataFrame:
        # TODO: Extremely hacky, fix later
        df, schema = load_csv("datasets/breast_cancer.csv")
        return df

        # num_records = 10
        # data = {
        #    "ints": np.random.randint(0, 100, size=num_records),
        #    "floats": np.random.randn(num_records),
        #    "dates": pd.date_range('2023-01-01', periods=num_records),
        #    "categories": np.random.choice(["a", "b", "c"], size=num_records)
        # }
        # return pd.DataFrame(data)