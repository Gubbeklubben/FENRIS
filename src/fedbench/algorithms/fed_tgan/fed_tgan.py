from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd
import numpy as np
import torch

from fedbench.core.algorithm import Aggregator, Synthesizer, Algorithm
from fedbench.core.data import load_csv
from fedbench.core.update import Update
from fedbench.core.logger import debug_calls

from fedbench.algorithms.fed_tgan.generator import Generator
from fedbench.algorithms.fed_tgan.discriminator import Discriminator


_some_state = {
    "s1": {"s11": 1, "s12": 2, "s13": 3, "nested_dict": {1: 1}},
    "s2": {"s21": 1, "s22": 2, "s23": 3, "nested_dict": {2: 2}},
}


def init_model(cfg: dict[str, Any]) -> tuple[Generator, Discriminator]:
    # Initialize generator and discriminator with provided config dict after these are implemented
    return


class FedTGAN(Algorithm):
    def __init__(
            self,
            batch_size: int = 32,
            learning_rate: float = 1e-2,    # starting point, adjust if needed
            fraction_evaluate: float = 0.5,
            num_server_rounds: int = 3,
            local_epochs: int = 3,
        ):

        # TODO error handling, in the same vein as below:
        if learning_rate <= 0 or learning_rate > 0.1:
            raise ValueError("Expecting 0 < learning_rate <= 0.1")

        self._cfg = {
            "batch-size": batch_size,
            "learning-rate": learning_rate,
            "fraction-evaluate": fraction_evaluate,
            "num-server-rounds": num_server_rounds,
            "local-epochs": local_epochs,
            "device": torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        }

    @classmethod
    def create_aggregator(cls) -> Aggregator:
        return FedTGANAggregator(cls._cfg)

    @classmethod
    def create_synthesizer(cls) -> Synthesizer:
        return FedTGANSynthesizer(cls._cfg)


class FedTGANAggregator(Aggregator):
    @debug_calls(__name__)
    def aggregate_init(self, replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-state"] = _some_state
        return update

    @debug_calls(__name__)
    def aggregate_train(
            self,
            replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-state"] = _some_state
        return update


class FedTGANSynthesizer(Synthesizer):
    @debug_calls(__name__)
    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        update = Update()
        update.objects["my-state"] = _some_state
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

        # num_records = 10
        # data = {
        #    "ints": np.random.randint(0, 100, size=num_records),
        #    "floats": np.random.randn(num_records),
        #    "dates": pd.date_range('2023-01-01', periods=num_records),
        #    "categories": np.random.choice(["a", "b", "c"], size=num_records)
        # }
        # return pd.DataFrame(data)