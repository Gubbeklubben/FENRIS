from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from fedbench.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fedbench.builtins.synthesizers.fed_tgan.bgm_transformer import BGMTransformer
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


def apply_activate(data: torch.Tensor, output_info: list[tuple[int, str]]) -> torch.Tensor:
    """Apply column-specific activations.

    For continuous columns: Apply tanh (maps to [-1, 1])
    For categorical columns: Apply Gumbel-Softmax (differentiable sampling)

    Parameters
    ----------
    data : torch.Tensor
        Raw generator output
    output_info : list[tuple[int, str]]
        List of (dimension, activation_type) per column from BGMTransformer

    Returns
    -------
    torch.Tensor
        Activated data
    """
    data_t = []
    st = 0
    for dim, activation in output_info:
        if activation == "tanh":
            ed = st + dim
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif activation == "softmax":
            ed = st + dim
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2, hard=False))
            st = ed
        else:
            raise ValueError(f"Unknown activation: {activation}")

    return torch.cat(data_t, dim=1)


@dataclass(frozen=True)
class _FedTGANArtifacts:
    cat_attrs: list[str]
    num_attrs: list[str]
    input_dim: int
    output_dim: int
    transformer: BGMTransformer
    output_info: list[tuple[int, str]]

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
            transformer=cast(BGMTransformer, objects["transformer"]),
            output_info=cast(list[tuple[int, str]], objects["output-info"]),
        )

    def encode(self) -> Payload:
        return Payload(
            objects={
                "objects": {
                    "transformer": self.transformer,
                    "output-info": self.output_info,
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
    def name(self) -> str:
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

        # Fit BGMTransformer
        transformer = BGMTransformer(n_clusters=10, eps=0.005)
        transformer.fit(dataset, cat_attrs, num_attrs)

        # Get output dimension and info from transformer
        output_dim = transformer.get_output_dim()
        output_info = transformer.get_output_info()
        input_dim = output_dim

        # Initialize models with correct dimensions
        generator = Generator(latent_dim=self._latent_dim, output_dim=output_dim)
        discriminator = Discriminator(input_dim=input_dim)

        # Pack both models into a single state_dict with prefixed keys
        packed_state: dict[str, torch.Tensor] = {}
        for k, v in generator.state_dict().items():
            packed_state[f"generator.{k}"] = v
        for k, v in discriminator.state_dict().items():
            packed_state[f"discriminator.{k}"] = v

        artifacts = _FedTGANArtifacts(
            cat_attrs=cat_attrs,
            num_attrs=num_attrs,
            input_dim=input_dim,
            output_dim=output_dim,
            transformer=transformer,
            output_info=output_info,
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

        input_dim = artifacts.input_dim
        output_dim = artifacts.output_dim
        transformer = artifacts.transformer
        output_info = artifacts.output_info

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

        # Transform data with BGMTransformer
        x = transformer.transform(data).astype(np.float32)

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
                fake_data_raw = generator(noise)
                fake_data = apply_activate(fake_data_raw, output_info)

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

        output_dim = artifacts.output_dim
        transformer = artifacts.transformer
        output_info = artifacts.output_info

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
            synthetic_data_raw = generator(noise)
            synthetic_data = apply_activate(synthetic_data_raw, output_info)
            synthetic_data_np = synthetic_data.cpu().numpy()

        # Inverse transform with BGMTransformer
        return transformer.inverse_transform(synthetic_data_np, sigmas=None)
