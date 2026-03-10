from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd
import numpy as np
import torch

from fedbench.core.algorithm import Aggregator, Synthesizer, Algorithm
from fedbench.core.data import load_csv, TableSchema
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
            max_batches: int = 100,
            learning_rate: float = 1e-2,
            fraction_evaluate: float = 0.5,
            num_server_rounds: int = 3,
            local_epochs: int = 3,
            latent_dim: int = 64,
        ):

        # TODO basic validation, in the same vein as below:
        if learning_rate <= 0 or learning_rate > 0.1:
            raise ValueError("Expecting 0 < learning_rate <= 0.1")

        self._cfg = {
            "batch-size": batch_size,
            "max-batches": max_batches,
            "learning-rate": learning_rate,
            "fraction-evaluate": fraction_evaluate,
            "num-server-rounds": num_server_rounds,
            "local-epochs": local_epochs,
            "latent-dim": latent_dim,
            "device": torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        }

    def create_aggregator(self) -> Aggregator:
        return FedTGANAggregator(self._cfg)

    def create_synthesizer(self) -> Synthesizer:
        return FedTGANSynthesizer(self._cfg)


class FedTGANAggregator(Aggregator):

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg: dict[str, Any] = cfg

    def configure_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int]) -> Iterable[tuple[int, Update]]:
        return ((client_id, Update()) for client_id in client_ids)

    def aggregate_init(self, replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-state"] = _some_state
        return update

    def aggregate_train(
            self,
            replies: Iterable[Update]) -> Update:
        update = Update()
        update.objects["my-state"] = _some_state
        return update


class FedTGANSynthesizer(Synthesizer):

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._max_batches = cfg["max-batches"]
        self._device = cfg["device"]

    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        update = Update()
        update.objects["my-state"] = _some_state
        return update

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