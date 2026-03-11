from collections.abc import Iterable
from typing import Any

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from fedbench.core.algorithm import Coordinator, SingleStepCoordinator, Synthesizer, Algorithm
from fedbench.core.data import TableSchema
from fedbench.core.update import Update

from fedbench.algorithms.fed_tgan.generator import Generator
from fedbench.algorithms.fed_tgan.discriminator import Discriminator


def split_cat_num(schema: TableSchema) -> tuple[list[str], list[str]]:
    """Split schema into categorical and numerical column names."""
    cat_attrs = [
        c.name
        for c in schema.columns
        if c.kind in ("categorical", "binary")
    ]
    num_attrs = [
        c.name
        for c in schema.columns
        if c.kind in ("continuous", "integer")
    ]
    return cat_attrs, num_attrs


def init_model(cfg: dict[str, Any]) -> tuple[Generator, Discriminator]:
    generator = Generator(
        latent_dim=cfg["latent_dim"],
        output_dim=cfg["output_dim"]
    )
    discriminator = Discriminator(
        input_dim=cfg["input_dim"]
    )

    return generator, discriminator


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

        if batch_size < 1:
            raise ValueError("Expecting batch_size >= 1.")
        if max_batches < 1:
            raise ValueError("Expecting max_batches >= 1.")
        if fraction_evaluate < 0 or fraction_evaluate > 1:
            raise ValueError("Expecting 0 <= fraction_evaluate <= 1.")
        if num_server_rounds < 1:
            raise ValueError("Expecting num_server_rounds >= 1.")
        if local_epochs < 1:
            raise ValueError("Expecting local_epochs >= 1.")
        if latent_dim < 1:
            raise ValueError("Expecting latent_dim >= 1.")
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

    def create_coordinator(self) -> Coordinator:
        return FedTGANCoordinator(self._cfg)

    def create_synthesizer(self) -> Synthesizer:
        return FedTGANSynthesizer(self._cfg)


class FedTGANCoordinator(SingleStepCoordinator):

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg: dict[str, Any] = cfg
        self._cat_attrs: list[str] | None = None
        self._num_attrs: list[str] | None = None
        self._preproc_objects: dict[str, Any] | None = None
        self._preproc_extras: dict[str, Any] | None = None
        self._state: dict[str, Any] | None = None

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"arrays": "torch"}

    @property
    def global_state(self) -> Update | None:
        if self._state is None:
            return None
        return self._create_update()

    def configure_fed_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int]) -> Iterable[tuple[int, Update]]:

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Split schema into categorical and numerical columns
        self._cat_attrs, self._num_attrs = split_cat_num(schema)

        # Send empty requests to all clients to collect vocabularies
        return ((client_id, Update()) for client_id in client_ids)

    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        # Collect vocabularies from all clients
        global_vocab: dict[str, set[str]] = {col: set() for col in self._cat_attrs}

        for _, reply in replies:
            if "vocab" in reply.extras:
                client_vocab = reply.extras["vocab"]
                for col, values in client_vocab.items():
                    global_vocab[col].update(values)

        # Create label encoders for each categorical column
        label_encoders: dict[str, LabelEncoder] = {}
        for col, values in global_vocab.items():
            if values:  # Only create encoder if there are values
                sorted_values = sorted(values)
                label_encoders[col] = LabelEncoder().fit(sorted_values)

        # Calculate dimensions for model initialization
        n_cat_features = len(self._cat_attrs)
        n_num_features = len(self._num_attrs)
        input_dim = output_dim = n_cat_features + n_num_features

        # Store preprocessing artifacts
        self._preproc_objects = {
            "label-encoders": label_encoders,
            "num-scaler": None,  # Will be set during first training round if needed
        }
        self._preproc_extras = {
            "cat-attrs": self._cat_attrs,
            "num-attrs": self._num_attrs,
            "input-dim": input_dim,
            "output-dim": output_dim,
        }

        # Initialize models with correct dimensions
        cfg_with_dims = self._cfg | {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "latent_dim": self._cfg["latent-dim"],
        }
        generator, discriminator = init_model(cfg_with_dims)

        self._state = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }

    def aggregate_train(
            self,
            replies: Iterable[tuple[int, Update]]) -> None:
        update = Update()
        update.objects["my-state"] = self._state
        return update

    def _create_update(self) -> Update:
        """Create Update with the current global state and preprocessing artifacts."""
        return Update(
            arrays={"arrays": self._state},
            objects={"preproc-objects": self._preproc_objects},
            extras={"preproc-extras": self._preproc_extras},
        )


class FedTGANSynthesizer(Synthesizer):

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._max_batches = cfg["max-batches"]
        self._device = cfg["device"]

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"arrays": "torch"}

    def fed_init(
            self,
            request: Update,
            seed: int,
            schema: TableSchema,
            data: pd.DataFrame) -> Update:
        """Extract unique categorical values from local data and return to coordinator."""

        cat_attrs, num_attrs = split_cat_num(schema)

        # Collect unique categorical values from this client's data
        vocab: dict[str, list[str]] = {}
        for col in cat_attrs:
            # Convert to string and get unique values
            unique_vals = data[col].astype(str).unique().tolist()
            vocab[col] = unique_vals

        # Return vocabulary to coordinator
        update = Update()
        update.extras["vocab"] = vocab
        return update

    def train(
            self,
            request: Update,
            data: pd.DataFrame) -> Update:

        update = Update()
        return update

    def sample(
            self,
            request: Update,
            num_rows: int,
            seed: int) -> pd.DataFrame:
        # TODO: Implement sampling
        raise NotImplementedError("Sampling not yet implemented")