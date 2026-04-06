from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame

from fedbench.builtins.coordinators.fed_tgan import ClientUpdate, GlobalState
from fedbench.builtins.synthesizers.fed_tgan.bgm_transformer import BGMTransformer
from fedbench.builtins.synthesizers.fed_tgan.conditional import Cond, cond_loss
from fedbench.builtins.synthesizers.fed_tgan.discriminator import Discriminator
from fedbench.builtins.synthesizers.fed_tgan.generator import Generator
from fedbench.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fedbench.core.data import TableSchema
from fedbench.core.logger import log_info
from fedbench.core.payload import ArraysTarget, Payload


def split_cat_num(schema: TableSchema) -> tuple[list[str], list[str]]:
    """Split schema into categorical and numerical column names."""
    cat_attrs = [c.name for c in schema.columns if c.kind in ("categorical", "binary")]
    num_attrs = [c.name for c in schema.columns if c.kind in ("continuous", "integer")]
    return cat_attrs, num_attrs


def compute_column_distributions(
    data: DataFrame, cat_attrs: list[str], num_attrs: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Compute categorical and continuous column distributions for table similarity.

    Ported from the original Fed-TGAN implementation:
    https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/distributed.py

    Parameters
    ----------
    data : DataFrame
        Raw client data (not transformed)
    cat_attrs : list[str]
        Categorical column names
    num_attrs : list[str]
        Numerical column names

    Returns
    -------
    tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]
        (categorical_distributions, numerical_distributions)
        - categorical_distributions: {column_name: {category: probability}}
        - numerical_distributions: {column_name: sample_values}
    """
    cat_distributions = {}
    num_distributions = {}

    # Compute categorical distributions (normalized value counts as dict)
    for col in cat_attrs:
        value_counts = data[col].value_counts(normalize=True)
        # Store as dictionary: {category: probability}
        cat_distributions[col] = {str(k): float(v) for k, v in value_counts.items()}

    # Store continuous column samples
    for col in num_attrs:
        num_distributions[col] = np.asarray(data[col].values, dtype=np.float32)

    return cat_distributions, num_distributions


def apply_activate(
    data: torch.Tensor, output_info: list[tuple[int, str]]
) -> torch.Tensor:
    """Apply column-specific activations.

    Ported from the original Fed-TGAN implementation:
    https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/synthesizers/ctgan.py

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


class Sampler:
    """Samples real data conditioned on a categorical column and category.

    Ported from the original Fed-TGAN implementation (ctgan.py Sampler class):
    https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/synthesizers/ctgan.py

    For each categorical column, stores the row indices for each category,
    enabling conditioned sampling to pair real data with fake data during training.
    This is the training-by-sampling mechanism that prevents mode collapse.
    """

    def __init__(self, data: np.ndarray, output_info: list[tuple[int, str]]) -> None:
        self._data = data
        self._n = len(data)
        # _model[col_idx][category_idx] -> array of row indices with that category
        self._model: list[list[np.ndarray]] = []

        st = 0
        for dim, activation in output_info:
            if activation == "tanh":
                st += dim
            elif activation == "softmax":
                ed = st + dim
                col_model = []
                for j in range(dim):
                    col_model.append(np.nonzero(data[:, st + j])[0])
                self._model.append(col_model)
                st = ed

    def sample(
        self,
        n: int,
        col: np.ndarray | None,
        opt: np.ndarray | None,
    ) -> np.ndarray:
        """Sample n rows, optionally conditioned on column/category indices.

        Parameters
        ----------
        n : int
            Number of rows to sample
        col : np.ndarray | None
            Column indices to condition on (one per sample), or None for random
        opt : np.ndarray | None
            Category indices within each column, or None for random

        Returns
        -------
        np.ndarray
            Sampled rows from the training data
        """
        if col is None:
            idx: np.ndarray = np.random.choice(np.arange(self._n), n)
            # noinspection PyUnnecessaryCast
            return cast(np.ndarray, self._data[idx])
        assert opt is not None
        idx = np.array([np.random.choice(self._model[c][o]) for c, o in zip(col, opt)])
        # noinspection PyUnnecessaryCast
        return cast(np.ndarray, self._data[idx])


def _calc_gradient_penalty(
    discriminator: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    pac: int = 10,
    lambda_: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP gradient penalty (Gulrajani et al. 2017).

    Ported from the original Fed-TGAN implementation (ctgan.py calc_gradient_penalty):
    https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/synthesizers/ctgan.py

    Interpolates between real and fake samples and penalizes discriminator
    gradients whose L2 norm deviates from 1, enforcing the Lipschitz constraint
    required for stable WGAN training.

    Parameters
    ----------
    discriminator : Discriminator
        The PacGAN discriminator
    real : torch.Tensor
        Real data (with conditional vectors concatenated), shape (batch, dim)
    fake : torch.Tensor
        Fake data (with conditional vectors concatenated), shape (batch, dim)
    device : torch.device
        Computation device
    pac : int
        PacGAN pack size (must match discriminator.pack)
    lambda_ : float
        Gradient penalty coefficient

    Returns
    -------
    torch.Tensor
        Scalar gradient penalty
    """
    alpha = torch.rand(real.size(0), 1, device=device)
    # Detach fake before interpolating so gradients only flow through interpolates
    interpolates = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Reshape for PacGAN: (batch/pac, pac*dim), then compute per-pack L2 norm
    gradient_penalty = (
        (gradients.view(-1, pac * real.size(1)).norm(2, dim=1) - 1) ** 2
    ).mean() * lambda_
    return gradient_penalty


@dataclass(frozen=True)
class _FedTGANArtifacts:
    cat_attrs: list[str]
    num_attrs: list[str]
    output_dim: int
    disc_input_dim: int  # output_dim + cond_dim; stored explicitly to avoid recomputing
    transformer: BGMTransformer
    output_info: list[tuple[int, str]]
    cond: Cond  # Stored for use during sampling to reproduce training data distribution

    # noinspection PyUnnecessaryCast
    @classmethod
    def decode(cls, payload: Payload) -> Self:
        objects = payload.objects["objects"]
        extras = payload.extras["extras"]
        return cls(
            cat_attrs=cast(list[str], extras["cat-attrs"]),
            num_attrs=cast(list[str], extras["num-attrs"]),
            output_dim=cast(int, extras["output-dim"]),
            disc_input_dim=cast(int, extras["disc-input-dim"]),
            transformer=cast(BGMTransformer, objects["transformer"]),
            output_info=cast(list[tuple[int, str]], objects["output-info"]),
            cond=cast(Cond, objects["cond"]),
        )

    def encode(self) -> Payload:
        return Payload(
            objects={
                "objects": {
                    "transformer": self.transformer,
                    "output-info": self.output_info,
                    "cond": self.cond,
                }
            },
            extras={
                "extras": {
                    "cat-attrs": self.cat_attrs,
                    "num-attrs": self.num_attrs,
                    "output-dim": self.output_dim,
                    "disc-input-dim": self.disc_input_dim,
                }
            },
        )


class FedTGAN(Synthesizer):
    def __init__(
        self,
        batch_size: int = 500,
        max_batches: int = 100,
        local_epochs: int = 5,
        learning_rate: float = 2e-4,
        latent_dim: int = 64,
    ) -> None:

        if batch_size < 1:
            raise ValueError("Expecting batch_size >= 1.")
        if batch_size % 10 != 0:
            raise ValueError("Expecting batch_size to be divisible by 10 (for PacGAN).")
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
        log_info(
            str(self),
            f"Initialized synthesizer with torch device {self._device.type.upper()}",
        )

    @property
    def name(self) -> str:
        return "fed_tgan"

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.TORCH

    @property
    def supports_coordinators(self) -> set[str]:
        return {"fed_tgan"}

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

        # Conditional dimension: sum of all softmax columns (categorical + BGM modes)
        cond_dim = sum(dim for dim, act in output_info if act == "softmax")

        # Discriminator input = data + conditional (ctgan.py line 353)
        disc_input_dim = output_dim + cond_dim

        # Build Cond from global data so sampling can reproduce the training
        # distribution
        x_global = transformer.transform(dataset).astype(np.float32)
        cond = Cond(x_global, output_info)

        # Generator input = latent + conditional (ctgan.py line 349)
        generator = Generator(
            latent_dim=self._latent_dim + cond_dim, output_dim=output_dim
        )
        discriminator = Discriminator(input_dim=disc_input_dim)

        # Pack both models into a single state_dict with prefixed keys
        packed_state: dict[str, torch.Tensor] = {}
        for k, v in generator.state_dict().items():
            packed_state[f"generator.{k}"] = v
        for k, v in discriminator.state_dict().items():
            packed_state[f"discriminator.{k}"] = v

        artifacts = _FedTGANArtifacts(
            cat_attrs=cat_attrs,
            num_attrs=num_attrs,
            output_dim=output_dim,
            disc_input_dim=disc_input_dim,
            transformer=transformer,
            output_info=output_info,
            cond=cond,
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

        output_dim = artifacts.output_dim
        disc_input_dim = artifacts.disc_input_dim
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

        # Initialize conditional vector generator and conditioned data sampler
        cond_generator = Cond(x, output_info)
        data_sampler = Sampler(x, output_info)

        # Calculate conditional dimension
        cond_dim = cond_generator.n_opt if cond_generator.n_col > 0 else 0

        # Initialize models
        generator = Generator(
            latent_dim=self._latent_dim + cond_dim, output_dim=output_dim
        )
        discriminator = Discriminator(input_dim=disc_input_dim)

        # Load weights from request
        generator.load_state_dict(generator_state)
        discriminator.load_state_dict(discriminator_state)

        # Move to device
        generator.to(self._device)
        discriminator.to(self._device)

        # Adam optimizer matching ctgan.py lines 355-356
        optimizer_generator = torch.optim.Adam(
            generator.parameters(), lr=self._learning_rate, betas=(0.5, 0.9)
        )
        optimizer_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=self._learning_rate, betas=(0.5, 0.9)
        )

        train_loss_discriminator = 0.0
        train_loss_generator = 0.0
        num_steps = 0

        steps_per_epoch = max(1, len(x) // self._batch_size)

        # Training loop ported from ctgan.py CTGANSynthesizer.fit (lines 362-433)
        # and distributed.py MDGANClient.train_model (lines 328-417)
        for _ in range(self._local_epochs):
            for _ in range(steps_per_epoch):
                # Sample conditional vectors for this step
                condvec = cond_generator.sample(self._batch_size)
                if condvec is not None:
                    c1, _, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    # Sample real data conditioned on same column/category (shuffled
                    # permutation avoids discriminator learning the pairing between
                    # fake and real samples) (ctgan.py lines 377-380)
                    perm = np.random.permutation(self._batch_size)
                    real_np = data_sampler.sample(
                        self._batch_size, col[perm], opt[perm]
                    )
                    real_data = torch.from_numpy(real_np.astype(np.float32)).to(
                        self._device
                    )
                    c2 = c1[perm]
                else:
                    c1 = torch.zeros(self._batch_size, 0, device=self._device)
                    c2 = c1
                    real_np = data_sampler.sample(self._batch_size, None, None)
                    real_data = torch.from_numpy(real_np.astype(np.float32)).to(
                        self._device
                    )

                # ===== Train Discriminator =====
                discriminator.train()

                # Generate fake data conditioned on c1
                noise = torch.randn(
                    self._batch_size, self._latent_dim, device=self._device
                )
                noise_c = torch.cat([noise, c1], dim=1) if c1.size(1) > 0 else noise
                fake_data_raw = generator(noise_c)
                fake_data = apply_activate(fake_data_raw, output_info)

                # Concatenate conditional vectors to both real and fake before
                # discriminator (ctgan.py lines 386-391)
                if c1.size(1) > 0:
                    fake_cat = torch.cat([fake_data, c1], dim=1)
                    real_cat = torch.cat([real_data, c2], dim=1)
                else:
                    fake_cat = fake_data
                    real_cat = real_data

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                # Wasserstein discriminator loss + gradient penalty
                # (ctgan.py lines 396-402)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = _calc_gradient_penalty(
                    discriminator,
                    real_cat,
                    fake_cat,
                    self._device,
                    pac=discriminator.pack,
                )

                optimizer_discriminator.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizer_discriminator.step()

                train_loss_discriminator += loss_d.item()

                # ===== Train Generator =====
                # Resample condvec for generator update (ctgan.py lines 404-413)
                condvec2 = cond_generator.sample(self._batch_size)
                if condvec2 is not None:
                    c1_g, m1_g, _, _ = condvec2
                    c1_g = torch.from_numpy(c1_g).to(self._device)
                    m1_g = torch.from_numpy(m1_g).to(self._device)
                else:
                    c1_g = torch.zeros(self._batch_size, 0, device=self._device)
                    m1_g = torch.zeros(self._batch_size, 0, device=self._device)

                generator.train()
                discriminator.eval()

                noise = torch.randn(
                    self._batch_size, self._latent_dim, device=self._device
                )
                noise_c = torch.cat([noise, c1_g], dim=1) if c1_g.size(1) > 0 else noise
                fake_data_raw = generator(noise_c)
                fake_data = apply_activate(fake_data_raw, output_info)

                # (ctgan.py lines 418-421)
                if c1_g.size(1) > 0:
                    y_fake_g = discriminator(torch.cat([fake_data, c1_g], dim=1))
                else:
                    y_fake_g = discriminator(fake_data)

                # Wasserstein generator loss + conditional cross-entropy (ctgan.py lines
                # 423-428)
                loss_g_adv = -torch.mean(y_fake_g)
                if condvec2 is not None and c1_g.size(1) > 0:
                    loss_g: torch.Tensor = loss_g_adv + cond_loss(
                        fake_data_raw, output_info, c1_g, m1_g
                    )
                else:
                    loss_g = loss_g_adv

                optimizer_generator.zero_grad()
                loss_g.backward()
                optimizer_generator.step()

                train_loss_generator += loss_g.item()
                num_steps += 1

                if num_steps >= self._max_batches:
                    break

            if num_steps >= self._max_batches:
                break

        # Pack both models into a single state_dict with prefixed keys
        packed_state: dict[str, torch.Tensor] = {}
        for k, v in generator.state_dict().items():
            packed_state[f"generator.{k}"] = v
        for k, v in discriminator.state_dict().items():
            packed_state[f"discriminator.{k}"] = v

        # Compute column distributions for table similarity
        cat_distributions, num_distributions = compute_column_distributions(
            data, artifacts.cat_attrs, artifacts.num_attrs
        )

        reply = ClientUpdate(
            state=packed_state,
            count=len(x),
            cat_distributions=cat_distributions,
            num_distributions=num_distributions,
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
        cond = artifacts.cond

        # Unpack generator state (only need generator for sampling)
        generator_state = {
            k.removeprefix("generator."): v
            for k, v in state.items()
            if k.startswith("generator.")
        }

        # Calculate conditional dimension
        cond_dim = sum(dim for dim, act in output_info if act == "softmax")

        # Initialize and load generator (with conditional dimension)
        generator = Generator(
            latent_dim=self._latent_dim + cond_dim, output_dim=output_dim
        )
        generator.load_state_dict(generator_state)
        generator.to(self._device)
        generator.eval()

        # Generate synthetic data in batches (to handle arbitrary num_rows).
        # Uses sample_original_training_data_prob() to match the reference
        # (ctgan.py CTGANSynthesizer.sample, which calls sample_zero()).
        all_chunks: list[np.ndarray] = []
        remaining = context.num_rows

        with torch.no_grad():
            while remaining > 0:
                batch = min(remaining, self._batch_size)

                noise = torch.randn(batch, self._latent_dim, device=self._device)

                # Sample conditional vectors from training data distribution
                condvec = cond.sample_original_training_data_prob(batch)
                if condvec is not None:
                    c = torch.from_numpy(condvec).to(self._device)
                    noise_c = torch.cat([noise, c], dim=1)
                else:
                    noise_c = noise

                synthetic_raw = generator(noise_c)
                synthetic = apply_activate(synthetic_raw, output_info)
                all_chunks.append(synthetic.cpu().numpy())
                remaining -= batch

        synthetic_data_np = np.concatenate(all_chunks, axis=0)[: context.num_rows]

        # Inverse transform with BGMTransformer
        return transformer.inverse_transform(synthetic_data_np, sigmas=None)
