import functools
from collections.abc import Iterable
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import Tensor

from fedbench.algorithms.fed_tgan.discriminator import Discriminator
from fedbench.algorithms.fed_tgan.generator import Generator
from fedbench.algorithms.fed_tgan.training import discriminator_step, generator_step
from fedbench.core.algorithm import (
    Algorithm,
    ComponentSpec,
    Coordinator,
    GlobalInitArtifacts,
    SingleStepCoordinator,
    Synthesizer,
    coordinator_spec,
    synthesizer_spec,
)
from fedbench.core.data import TableSchema
from fedbench.core.logger import ELBOW, log_warning
from fedbench.core.update import Update


def split_cat_num(schema: TableSchema) -> tuple[list[str], list[str]]:
    """Split schema into categorical and numerical column names."""
    cat_attrs = [c.name for c in schema.columns if c.kind in ("categorical", "binary")]
    num_attrs = [c.name for c in schema.columns if c.kind in ("continuous", "integer")]
    return cat_attrs, num_attrs


class FedTGAN(Algorithm):
    def __init__(
        self,
        batch_size: int = 32,
        max_batches: int = 100,
        learning_rate: float = 1e-2,
        fraction_evaluate: float = 0.5,
        num_server_rounds: int = 5,
        local_epochs: int = 5,
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
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # Create factory function for synthesizer
        self._synth_factory: Callable[[], Synthesizer] = functools.partial(
            FedTGANSynthesizer,
            batch_size=batch_size,
            max_batches=max_batches,
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            local_epochs=local_epochs,
            device=self._cfg["device"],
        )

    @property
    def coordinator_spec(self) -> ComponentSpec[Coordinator]:
        return coordinator_spec(
            FedTGANCoordinator, {"initial-state": "torch", "state": "torch"}
        )

    @property
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        return synthesizer_spec(self._synth_factory, {"state": "torch"})

    def global_init(
        self, seed: int, schema: TableSchema, dataset: pd.DataFrame
    ) -> GlobalInitArtifacts | None:
        """Initialize preproc and models with access to full dataset and config."""

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Split schema into categorical and numerical columns
        cat_attrs, num_attrs = split_cat_num(schema)

        # Build global vocabulary for categorical columns
        vocab_classes: set[str] = set()
        for col in cat_attrs:
            unique_vals = dataset[col].astype(str).unique()
            vocab_classes.update(unique_vals)

        # Create label encoder with full vocabulary
        vocab_sorted = sorted(vocab_classes)
        label_encoder = LabelEncoder().fit(vocab_sorted)

        # Calculate dimensions
        n_cat_features = len(cat_attrs)
        n_num_features = len(num_attrs)
        input_dim = output_dim = n_cat_features + n_num_features

        # Use the configured latent_dim from algorithm initialization
        latent_dim = self._cfg["latent-dim"]

        # Initialize models with correct dimensions
        generator = Generator(latent_dim=latent_dim, output_dim=output_dim)
        discriminator = Discriminator(input_dim=input_dim)

        # Pack both models into a single state_dict with prefixed keys
        packed_state: dict[str, torch.Tensor] = {}
        for k, v in generator.state_dict().items():
            packed_state[f"generator.{k}"] = v
        for k, v in discriminator.state_dict().items():
            packed_state[f"discriminator.{k}"] = v

        # Fit scaler on numerical features
        num_scaler = None
        if num_attrs:
            num_scaler = MinMaxScaler()
            num_scaler.fit(dataset[num_attrs].values)

        # Prepare artifacts for synthesizers
        synth_artifacts = Update()
        synth_artifacts.objects["preproc-objects"] = {
            "label-encoders": {col: label_encoder for col in cat_attrs},
            "num-scaler": num_scaler,
        }
        synth_artifacts.extras["preproc-extras"] = {
            "cat-attrs": cat_attrs,
            "num-attrs": num_attrs,
            "input-dim": input_dim,
            "output-dim": output_dim,
            "latent-dim": latent_dim,
        }

        # Coordinator receives both model state and preprocessing artifacts
        # to pass preprocessing to synthesizers in each round
        coord_artifacts = Update(arrays={"initial-state": packed_state})
        coord_artifacts.objects["preproc-objects"] = synth_artifacts.objects[
            "preproc-objects"
        ]
        coord_artifacts.extras["preproc-extras"] = synth_artifacts.extras[
            "preproc-extras"
        ]

        return GlobalInitArtifacts(
            coordinator=coord_artifacts,
            synthesizer=synth_artifacts,
        )


class FedTGANCoordinator(SingleStepCoordinator):
    def __init__(self) -> None:
        self._state: dict[str, torch.Tensor] | None = None
        self._preproc_objects: dict[str, Any] | None = None
        self._preproc_extras: dict[str, Any] | None = None

    @property
    def global_state(self) -> Update | None:
        if self._state is None:
            return None
        return self._create_update()

    def attach_global_init_artifacts(self, artifacts: Update) -> None:
        """Receive initial state and preprocessing artifacts from global_init."""
        self._state = cast(dict[str, Tensor], artifacts.arrays["initial-state"])
        self._preproc_objects = artifacts.objects["preproc-objects"]
        self._preproc_extras = artifacts.extras["preproc-extras"]

    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        """Aggregate generator, discriminator using FedAvg weighted by num_samples."""
        if not replies:
            raise ValueError("No replies, can not aggregate.")

        num_samples: list[int] = []
        state_dicts: list[dict[str, Tensor]] = []

        for _, reply in replies:
            # noinspection PyUnnecessaryCast
            num_samples.append(cast(int, reply.metrics["metrics"]["num-samples"]))
            # noinspection PyUnnecessaryCast
            state_dicts.append(cast(dict[str, Tensor], reply.arrays["state"]))

        total = sum(num_samples)
        if total <= 0:
            log_warning(str(self), f"Total number of samples: {total}")
            log_warning("", f"\t{ELBOW} Skipping aggregation.")
            return

        weights = tuple(float(n) / total for n in num_samples)
        keys = tuple(state_dicts[0].keys())
        aggr_state: dict[str, Tensor] = {}

        with torch.no_grad():
            for key in keys:
                result: Tensor | None = None

                for state_dict, weight in zip(state_dicts, weights, strict=True):
                    tensor = state_dict[key].detach().cpu()
                    if result is None:
                        result = tensor * weight
                    else:
                        result = result + tensor * weight

                aggr_state[key] = result

        self._state = aggr_state

    def _create_update(self) -> Update:
        """Create Update with the current global state and preprocessing artifacts."""
        # _state is guaranteed non-None when this is called from global_state property
        state = cast(dict[str, Tensor], self._state)
        update = Update(arrays={"state": state})

        # Include preprocessing artifacts if available
        if self._preproc_objects is not None:
            update.objects["preproc-objects"] = self._preproc_objects
        if self._preproc_extras is not None:
            update.extras["preproc-extras"] = self._preproc_extras

        return update


class FedTGANSynthesizer(Synthesizer):
    def __init__(
        self,
        batch_size: int,
        max_batches: int,
        learning_rate: float,
        latent_dim: int,
        local_epochs: int,
        device: torch.device,
    ) -> None:
        self._batch_size = batch_size
        self._max_batches = max_batches
        self._learning_rate = learning_rate
        self._latent_dim = latent_dim
        self._local_epochs = local_epochs
        self._device = device

        # Will be populated by attach_global_init_artifacts
        self._preproc_objects: dict[str, Any] | None = None
        self._preproc_extras: dict[str, Any] | None = None

    def attach_global_init_artifacts(self, artifacts: Update) -> None:
        """Receive preprocessing artifacts from global_init."""
        self._preproc_objects = artifacts.objects["preproc-objects"]
        self._preproc_extras = artifacts.extras["preproc-extras"]

    def train(self, request: Update, data: pd.DataFrame) -> Update:
        """Train Generator and Discriminator on local data, alternating GAN training."""

        # Extract preprocessing artifacts and model states from request
        # noinspection PyUnnecessaryCast
        received_state = cast(dict[str, Tensor], request.arrays["state"])
        preproc_objects = request.objects["preproc-objects"]
        preproc_extras = request.extras["preproc-extras"]

        # noinspection PyUnnecessaryCast
        cat_attrs = cast(list[str], preproc_extras["cat-attrs"])
        # noinspection PyUnnecessaryCast
        num_attrs = cast(list[str], preproc_extras["num-attrs"])
        # noinspection PyUnnecessaryCast
        input_dim = cast(int, preproc_extras["input-dim"])
        # noinspection PyUnnecessaryCast
        output_dim = cast(int, preproc_extras["output-dim"])
        label_encoders = preproc_objects["label-encoders"]

        # Unpack generator and discriminator state_dicts from received state
        generator_state = {
            k.removeprefix("generator."): v
            for k, v in received_state.items()
            if k.startswith("generator.")
        }
        discriminator_state = {
            k.removeprefix("discriminator."): v
            for k, v in received_state.items()
            if k.startswith("discriminator.")
        }

        # Preprocess data: label-encode categoricals, concatenate with numericals
        processed_data = []

        # Encode categorical columns
        for col in cat_attrs:
            if col in label_encoders:
                encoded = label_encoders[col].transform(data[col].astype(str))
                processed_data.append(encoded.reshape(-1, 1))

        # Add numerical columns (scaled)
        num_scaler = preproc_objects["num-scaler"]
        if num_attrs and num_scaler is not None:
            num_data = data[num_attrs].values
            num_scaled = num_scaler.transform(num_data)
            processed_data.append(num_scaled)
        elif num_attrs:
            # Fallback if no scaler (shouldn't happen)
            num_data = data[num_attrs].values
            processed_data.append(num_data)

        # Concatenate all features
        if processed_data:
            X = np.hstack(processed_data).astype(np.float32)
        else:
            raise ValueError("No data to train on")

        # Convert to torch tensor and create DataLoader
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True
        )

        # Initialize models
        generator = Generator(latent_dim=self._latent_dim, output_dim=output_dim)
        discriminator = Discriminator(input_dim=input_dim)

        # Load weights from request
        generator.load_state_dict(generator_state)
        discriminator.load_state_dict(discriminator_state)

        # Move to device
        generator.to(self._device)
        discriminator.to(self._device)

        # Create optimizers
        optimizer_generator = torch.optim.SGD(
            generator.parameters(), lr=self._learning_rate, momentum=0.9
        )
        optimizer_discriminator = torch.optim.SGD(
            discriminator.parameters(), lr=self._learning_rate, momentum=0.9
        )

        train_loss_discriminator = 0.0
        train_loss_generator = 0.0
        num_batches = 0

        # Training loop
        for _ in range(self._local_epochs):
            for (real_data_batch,) in dataloader:
                real_data = real_data_batch.to(self._device)

                # Generate synthetic data
                noise = torch.randn(
                    real_data.size(0), self._latent_dim, device=self._device
                )
                fake_data = generator(noise)

                # Train discriminator on combined data
                train_loss_discriminator += discriminator_step(
                    discriminator,
                    real_data,
                    fake_data.detach(),
                    optimizer_discriminator,
                    self._device,
                )

                # Train the generator
                noise_g = torch.randn(
                    real_data.size(0), self._latent_dim, device=self._device
                )
                train_loss_generator += generator_step(
                    generator, discriminator, noise_g, optimizer_generator, self._device
                )

                num_batches += 1

                # Stop if max_batches reached
                if num_batches >= self._max_batches:
                    break

            if num_batches >= self._max_batches:
                break

        # Pack both models into a single state_dict with prefixed keys
        packed_state: dict[str, torch.Tensor] = {}
        for k, v in generator.state_dict().items():
            packed_state[f"generator.{k}"] = v
        for k, v in discriminator.state_dict().items():
            packed_state[f"discriminator.{k}"] = v

        # Return update with both models
        reply = Update()
        reply.arrays["state"] = packed_state
        reply.metrics["metrics"] = {
            "loss-discriminator": train_loss_discriminator / num_batches
            if num_batches > 0
            else 0.0,
            "loss-generator": train_loss_generator / num_batches
            if num_batches > 0
            else 0.0,
            "num-samples": len(dataset),
        }

        return reply

    def sample(self, request: Update, num_rows: int, seed: int) -> pd.DataFrame:
        """Generate synthetic data using the trained generator."""

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Extract model state and preprocessing artifacts from request
        # noinspection PyUnnecessaryCast
        packed_state = cast(dict[str, Tensor], request.arrays["state"])
        preproc_objects = request.objects["preproc-objects"]
        preproc_extras = request.extras["preproc-extras"]

        # noinspection PyUnnecessaryCast
        cat_attrs = cast(list[str], preproc_extras["cat-attrs"])
        # noinspection PyUnnecessaryCast
        num_attrs = cast(list[str], preproc_extras["num-attrs"])
        # noinspection PyUnnecessaryCast
        output_dim = cast(int, preproc_extras["output-dim"])
        # noinspection PyUnnecessaryCast
        latent_dim = cast(int, preproc_extras["latent-dim"])
        label_encoders = preproc_objects["label-encoders"]

        # Unpack generator state (only need generator for sampling)
        generator_state = {
            k.removeprefix("generator."): v
            for k, v in packed_state.items()
            if k.startswith("generator.")
        }

        # Initialize and load generator
        generator = Generator(latent_dim=latent_dim, output_dim=output_dim)
        generator.load_state_dict(generator_state)
        generator.to(self._device)
        generator.eval()

        # Generate synthetic data
        with torch.no_grad():
            noise = torch.randn(num_rows, latent_dim, device=self._device)
            synthetic_data = generator(noise).cpu().numpy()

        # Reverse preproc: decode categorical features and extract numerical features
        decoded_data = {}
        n_cat_features = len(cat_attrs)

        # Decode categorical columns
        for i, col in enumerate(cat_attrs):
            if col in label_encoders:
                # Round to nearest integer for categorical encoding
                encoded_values = np.round(synthetic_data[:, i]).astype(int)
                # Clip to valid range
                encoded_values = np.clip(
                    encoded_values, 0, len(label_encoders[col].classes_) - 1
                )
                # Inverse transform to get original categorical values
                decoded_data[col] = label_encoders[col].inverse_transform(
                    encoded_values
                )

        # Extract and inverse scale numerical columns
        num_scaler = preproc_objects["num-scaler"]
        if num_attrs and num_scaler is not None:
            # Extract numerical columns
            num_synthetic = synthetic_data[:, n_cat_features : n_cat_features + len(num_attrs)]
            # Inverse transform to get back original scale
            num_original = num_scaler.inverse_transform(num_synthetic)
            # Add to decoded data
            for i, col in enumerate(num_attrs):
                decoded_data[col] = num_original[:, i]
        elif num_attrs:
            # Fallback if no scaler (shouldn't happen)
            for i, col in enumerate(num_attrs):
                decoded_data[col] = synthetic_data[:, n_cat_features + i]

        return pd.DataFrame(decoded_data)
