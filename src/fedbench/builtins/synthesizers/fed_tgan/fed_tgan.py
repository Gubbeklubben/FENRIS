from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from fedbench.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fedbench.builtins.synthesizers.fed_tgan.discriminator import Discriminator
from fedbench.builtins.synthesizers.fed_tgan.generator import Generator
from fedbench.builtins.synthesizers.fed_tgan.training import (
    discriminator_step,
    generator_step,
)
from fedbench.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fedbench.core.data import TableSchema
from fedbench.core.payload import ArraysTarget, Payload


def split_cat_num(schema: TableSchema) -> tuple[list[str], list[str]]:
    """Split schema into categorical and numerical column names."""
    cat_attrs = [c.name for c in schema.columns if c.kind in ("categorical", "binary")]
    num_attrs = [c.name for c in schema.columns if c.kind in ("continuous", "integer")]
    return cat_attrs, num_attrs


@dataclass(frozen=True)
class _FedTGANArtifacts:
    cat_attrs: list[str]
    num_attrs: list[str]
    input_dim: int
    output_dim: int
    cat_max_values: Mapping[str, int]
    label_encoders: Mapping[str, LabelEncoder]
    num_scaler: MinMaxScaler

    # noinspection PyUnnecessaryCast
    @classmethod
    def decode(cls, payload: Payload) -> Self:
        objects = payload.objects["objects"]
        extras = payload.extras["extras"]
        return cls(
            cat_attrs=cast(list[str], extras["cat-attrs"]),
            num_attrs=cast(list[str], extras["num-attrs"]),
            input_dim=cast(int, extras["input-dim"]),
            output_dim=cast(int, extras["output-dim"]),
            cat_max_values=cast(Mapping[str, int], objects["cat-max-values"]),
            label_encoders=cast(Mapping[str, LabelEncoder], objects["label-encoders"]),
            num_scaler=cast(MinMaxScaler, objects["num-scaler"]),
        )

    def encode(self) -> Payload:
        return Payload(
            objects={
                "objects": {
                    "cat-max-values": self.cat_max_values,
                    "label-encoders": self.label_encoders,
                    "num-scaler": self.num_scaler,
                }
            },
            extras={
                "extras": {
                    "cat-attrs": self.cat_attrs,
                    "num-attrs": self.num_attrs,
                    "input-dim": self.input_dim,
                    "output-dim": self.output_dim,
                }
            },
        )


class FedTGAN(Synthesizer):
    def __init__(
        self,
        batch_size: int = 32,
        max_batches: int = 100,
        local_epochs: int = 5,
        learning_rate: float = 1e-2,
        latent_dim: int = 64,
    ) -> None:

        if batch_size < 1:
            raise ValueError("Expecting batch_size >= 1.")
        if max_batches < 1:
            raise ValueError("Expecting max_batches >= 1.")
        if local_epochs < 1:
            raise ValueError("Expecting local_epochs >= 1.")
        if learning_rate <= 0 or learning_rate > 0.1:
            raise ValueError("Expecting 0 < learning_rate <= 0.1")
        if latent_dim < 1:
            raise ValueError("Expecting latent_dim >= 1.")

        self._batch_size = batch_size
        self._max_batches = max_batches
        self._learning_rate = learning_rate
        self._latent_dim = latent_dim
        self._local_epochs = local_epochs
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def id(self) -> str:
        return "fed_tgan"

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.TORCH

    @property
    def supports_coordinators(self) -> set[str]:
        return {"fedavg"}

    def global_init(
        self, dataset: DataFrame, context: GlobalInitContext
    ) -> GlobalInitArtifacts:

        np.random.seed(context.seed)
        torch.manual_seed(context.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(context.seed)

        # Split schema into categorical and numerical columns
        cat_attrs, num_attrs = split_cat_num(context.schema)

        # Create separate label encoder for each categorical column
        # and track max encoded value for normalization
        label_encoders = {}
        cat_max_values = {}
        for col in cat_attrs:
            unique_vals = sorted(dataset[col].astype(str).unique())
            le = LabelEncoder()
            le.fit(unique_vals)
            label_encoders[col] = le
            # Max encoded value = number of classes - 1
            cat_max_values[col] = len(unique_vals) - 1

        # Calculate dimensions
        n_cat_features = len(cat_attrs)
        n_num_features = len(num_attrs)
        input_dim = output_dim = n_cat_features + n_num_features

        # Initialize models with correct dimensions
        generator = Generator(latent_dim=self._latent_dim, output_dim=output_dim)
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

        artifacts = _FedTGANArtifacts(
            cat_attrs=cat_attrs,
            num_attrs=num_attrs,
            input_dim=input_dim,
            output_dim=output_dim,
            cat_max_values=cat_max_values,
            label_encoders=label_encoders,
            num_scaler=num_scaler,
        )
        return GlobalInitArtifacts(
            coordinator=GlobalState(packed_state).encode(),
            synthesizer=artifacts.encode(),
        )

    def train(
        self, request: Payload, data: DataFrame, context: TrainContext
    ) -> Payload:

        # noinspection PyUnnecessaryCast
        state = cast(dict[str, torch.Tensor], GlobalState.decode(request).state)
        if context.global_init_artifacts is None:
            raise RuntimeError("Missing preprocessing artifacts.")

        artifacts = _FedTGANArtifacts.decode(context.global_init_artifacts)

        cat_attrs = artifacts.cat_attrs
        num_attrs = artifacts.num_attrs
        input_dim = artifacts.input_dim
        output_dim = artifacts.output_dim
        cat_max_values = artifacts.cat_max_values
        label_encoders = artifacts.label_encoders
        num_scaler = artifacts.num_scaler

        # Unpack generator and discriminator state_dicts from received state
        generator_state = {
            k.removeprefix("generator."): v
            for k, v in state.items()
            if k.startswith("generator.")
        }
        discriminator_state = {
            k.removeprefix("discriminator."): v
            for k, v in state.items()
            if k.startswith("discriminator.")
        }

        # Preprocess data: label-encode categoricals, concatenate with numericals
        processed_data = []

        # Encode and normalize categorical columns
        for col in cat_attrs:
            if col in label_encoders:
                encoded = label_encoders[col].transform(data[col].astype(str))
                # Normalize to [0, 1] by dividing by max value
                max_val = cat_max_values[col]
                if max_val > 0:
                    normalized = encoded / max_val
                else:
                    normalized = encoded  # Single-class column, stays 0
                processed_data.append(normalized.reshape(-1, 1))

        # Add numerical columns (scaled)
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
            x = np.hstack(processed_data).astype(np.float32)
        else:
            raise ValueError("No data to train on")

        # Convert to torch tensor and create DataLoader
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x))
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

        reply = ClientUpdate(
            state=packed_state,
            count=len(dataset),
        )
        return reply.encode()

    def sample(self, request: Payload, context: SampleContext) -> DataFrame:
        np.random.seed(context.seed)
        torch.manual_seed(context.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(context.seed)

        # noinspection PyUnnecessaryCast
        state = cast(dict[str, torch.Tensor], GlobalState.decode(request).state)
        if context.global_init_artifacts is None:
            raise RuntimeError("Missing preprocessing artifacts.")

        artifacts = _FedTGANArtifacts.decode(context.global_init_artifacts)

        cat_attrs = artifacts.cat_attrs
        num_attrs = artifacts.num_attrs
        output_dim = artifacts.output_dim
        cat_max_values = artifacts.cat_max_values
        label_encoders = artifacts.label_encoders
        num_scaler = artifacts.num_scaler

        # Unpack generator state (only need generator for sampling)
        generator_state = {
            k.removeprefix("generator."): v
            for k, v in state.items()
            if k.startswith("generator.")
        }

        # Initialize and load generator
        generator = Generator(latent_dim=self._latent_dim, output_dim=output_dim)
        generator.load_state_dict(generator_state)
        generator.to(self._device)
        generator.eval()

        # Generate synthetic data
        with torch.no_grad():
            noise = torch.randn(context.num_rows, self._latent_dim, device=self._device)
            synthetic_data = generator(noise).cpu().numpy()

        # Reverse preproc: decode categorical features and extract numerical features
        decoded_data = {}
        n_cat_features = len(cat_attrs)

        # Decode categorical columns
        for i, col in enumerate(cat_attrs):
            if col in label_encoders:
                # Denormalize from [0, 1] back to integer range
                max_val = cat_max_values[col]
                denormalized = synthetic_data[:, i] * max_val
                # Round to nearest integer for categorical encoding
                encoded_values = np.round(denormalized).astype(int)
                # Clip to valid range
                encoded_values = np.clip(encoded_values, 0, max_val)
                # Inverse transform to get original categorical values
                decoded_data[col] = label_encoders[col].inverse_transform(
                    encoded_values
                )

        if num_attrs and num_scaler is not None:
            # Extract numerical columns
            num_synthetic = synthetic_data[
                :, n_cat_features : n_cat_features + len(num_attrs)
            ]
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
