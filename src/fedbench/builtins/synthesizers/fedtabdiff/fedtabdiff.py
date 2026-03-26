from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Self, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbench.builtins.coordinators.fedavg import ClientUpdate, GlobalState
from fedbench.builtins.synthesizers.fedtabdiff.diffuser import Diffuser
from fedbench.builtins.synthesizers.fedtabdiff.mlpsynthesizer import MLPSynthesizer
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
    cat_attrs = [
        c.name  # nofmt
        for c in schema.columns
        if c.kind in ("categorical", "binary")
    ]
    num_attrs = [
        c.name  # nofmt
        for c in schema.columns
        if c.kind in ("continuous", "integer")
    ]
    return cat_attrs, num_attrs


@dataclass(frozen=True)
class _FedTabDiffArtifacts:
    cat_attrs: list[str]
    num_attrs: list[str]
    n_cat_tokens: int
    cat_dim: int
    encoded_dim: int
    vocab_per_attr: dict[str, set[int]]
    num_scaler: QuantileTransformer
    label_encoder: LabelEncoder

    # noinspection PyUnnecessaryCast
    @classmethod
    def decode(cls, payload: Payload) -> Self:
        objects = payload.objects["objects"]
        extras = payload.extras["extras"]
        return cls(
            cat_attrs=cast(list[str], extras["cat-attrs"]),
            num_attrs=cast(list[str], extras["num-attrs"]),
            n_cat_tokens=cast(int, extras["n-cat-tokens"]),
            cat_dim=cast(int, extras["cat-dim"]),
            encoded_dim=cast(int, extras["encoded-dim"]),
            vocab_per_attr=objects["vocab-per-attr"],
            num_scaler=cast(QuantileTransformer, objects["num-scaler"]),
            label_encoder=cast(LabelEncoder, objects["label-encoder"]),
        )

    def encode(self) -> Payload:
        return Payload(
            objects={
                "objects": {
                    "vocab-per-attr": self.vocab_per_attr,
                    "num-scaler": self.num_scaler,
                    "label-encoder": self.label_encoder,
                }
            },
            extras={
                "extras": {
                    "cat-attrs": self.cat_attrs,
                    "num-attrs": self.num_attrs,
                    "n-cat-tokens": self.n_cat_tokens,
                    "cat-dim": self.cat_dim,
                    "encoded-dim": self.encoded_dim,
                }
            },
        )


class FedTabDiff(Synthesizer):
    def __init__(
        self,
        batch_size: int = 128,
        max_batches: int = 10,
        n_cat_emb: int = 2,
        learning_rate: float = 1e-4,
        mlp_layers: list[int] | None = None,
        activation: str = "lrelu",
        diffusion_steps: int = 500,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 0.02,
        scheduler: Literal["linear", "quad"] = "linear",
    ) -> None:

        # Note: AI used to suggest appropriate ranges for hard fail. There
        # are values well within ranges that might still make the algorithm
        # blow up.
        if n_cat_emb < 1:
            raise ValueError("Expecting n_cat_emb >= 1.")

        if learning_rate <= 0 or learning_rate > 0.1:
            raise ValueError("Expecting 0 < learning_rate <= 0.1")

        if mlp_layers is None:
            mlp_layers = [512, 512]

        if not mlp_layers:
            raise ValueError("Expecting non-empty mlp_layers.")

        for value in mlp_layers:
            if not isinstance(value, int):
                raise ValueError("Expecting int sequence mlp_layers.")

        if diffusion_steps < 1:
            raise ValueError("Expecting diffusion_steps >= 1.")

        if diffusion_beta_start <= 0:
            raise ValueError("Expecting diffusion_beta_start > 0.")

        if diffusion_beta_end >= 1:
            raise ValueError("Expecting diffusion_beta_end < 1.")

        if diffusion_beta_start >= diffusion_beta_end:
            raise ValueError("Expecting diffusion_beta_start < diffusion_beta_end")

        self._batch_size = batch_size
        self._max_batches = max_batches
        self._learning_rate = learning_rate
        self._n_cat_emb = n_cat_emb
        self._mlp_layers = mlp_layers
        self._activation = activation
        self._diffusion_steps = diffusion_steps
        self._diffusion_beta_start = diffusion_beta_start
        self._diffusion_beta_end = diffusion_beta_end
        self._scheduler = scheduler
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "fedtabdiff"

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

        cat_attrs, num_attrs = split_cat_num(context.schema)
        prefix_columns(dataset, cat_attrs)

        num_scaler = QuantileTransformer(
            n_quantiles=len(dataset),
            output_distribution="normal",
            random_state=context.seed,
        )
        num_scaler.fit(dataset[num_attrs].values)

        vocab_classes = sorted(np.unique(dataset[cat_attrs]))
        label_encoder = LabelEncoder().fit(vocab_classes)
        cat_scaled = dataset[cat_attrs].apply(label_encoder.transform)

        vocab_per_attr = {attr: set(cat_scaled[attr]) for attr in cat_attrs}
        n_cat_tokens = len(vocab_classes)
        cat_dim = self._n_cat_emb * len(cat_attrs)
        encoded_dim = cat_dim + len(num_attrs)

        artifacts = _FedTabDiffArtifacts(
            cat_attrs=cat_attrs,
            num_attrs=num_attrs,
            n_cat_tokens=n_cat_tokens,
            cat_dim=cat_dim,
            encoded_dim=encoded_dim,
            vocab_per_attr=vocab_per_attr,
            num_scaler=num_scaler,
            label_encoder=label_encoder,
        )
        mlp_synth = self._create_mlp_synth(encoded_dim, n_cat_tokens)

        return GlobalInitArtifacts(
            coordinator=GlobalState(mlp_synth.state_dict()).encode(),
            synthesizer=artifacts.encode(),
        )

    def train(
        self, request: Payload, data: DataFrame, context: TrainContext
    ) -> Payload:

        state = GlobalState.decode(request).state
        if context.global_init_artifacts is None:
            raise RuntimeError("Missing preprocessing artifacts.")

        artifacts = _FedTabDiffArtifacts.decode(context.global_init_artifacts)

        prefix_columns(data, artifacts.cat_attrs)
        cat_scaled = data[artifacts.cat_attrs].apply(artifacts.label_encoder.transform)
        num_scaled = artifacts.num_scaler.transform(data[artifacts.num_attrs].values)

        tensor_dataset = TensorDataset(
            torch.tensor(cat_scaled.values, dtype=torch.long),
            torch.tensor(num_scaled, dtype=torch.float),
        )
        torch_loader = DataLoader(
            tensor_dataset, batch_size=self._batch_size, shuffle=True
        )

        # Adapt unsupervised fedbench training to orig alg loop
        def loader() -> Iterator[tuple[Tensor, Tensor, Tensor | None]]:
            for cat, num in torch_loader:
                yield cat, num, None

        mlp_synth = self._create_mlp_synth(
            artifacts.encoded_dim, artifacts.n_cat_tokens
        )
        mlp_synth.load_state_dict(state)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, mlp_synth.parameters()),
            lr=self._learning_rate,
        )
        # set network in training mode
        mlp_synth.train()
        mlp_synth.to(self._device)

        diffuser = self._create_diffuser()
        loss_fnc = nn.MSELoss()

        num_samples = 0
        for idx, (batch_cat, batch_num, batch_y) in enumerate(loader()):
            num_samples += len(batch_cat)  # len operates on 1st tensor dim
            # move batch to device
            batch_cat = batch_cat.to(self._device)
            batch_num = batch_num.to(self._device)
            if batch_y is not None:
                batch_y = batch_y.to(self._device)

            # sample timestamps t
            timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])
            # get cat embeddings
            batch_cat_emb = mlp_synth.embed_categorical(x_cat=batch_cat)
            # concat cat & num
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)
            # add noise
            batch_noise_t, noise_t = diffuser.add_gauss_noise(
                x_num=batch_cat_num,
                timesteps=timesteps,
            )
            # conduct forward encoder/decoder pass
            predicted_noise = mlp_synth(
                x=batch_noise_t,
                timesteps=timesteps,
                label=batch_y,
            )
            # compute train loss
            train_losses = loss_fnc(
                input=noise_t,
                target=predicted_noise,
            )
            # reset encoder and decoder gradients
            optimizer.zero_grad()
            # run error back-propagation
            train_losses.backward()
            # optimize encoder and decoder parameters
            optimizer.step()

            if idx + 1 >= self._max_batches:
                break

        reply = ClientUpdate(
            state=mlp_synth.state_dict(),
            count=num_samples,
        )
        return reply.encode()

    def sample(self, request: Payload, context: SampleContext) -> DataFrame:
        torch.manual_seed(context.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(context.seed)

        state = GlobalState.decode(request).state
        if context.global_init_artifacts is None:
            raise RuntimeError("Missing preprocessing artifacts.")

        artifacts = _FedTabDiffArtifacts.decode(context.global_init_artifacts)

        mlp_synth = self._create_mlp_synth(
            artifacts.encoded_dim, artifacts.n_cat_tokens
        )
        mlp_synth.load_state_dict(state)
        mlp_synth.to(self._device)

        diffuser = self._create_diffuser()

        tensor = self._generate_samples(
            mlp_synth=mlp_synth,
            diffuser=diffuser,
            artifacts=artifacts,
            n_samples=context.num_rows,
            label=None,
        )
        return self._decode_samples(
            samples=tensor, embeddings=mlp_synth.get_embeddings(), artifacts=artifacts
        )

    def _create_mlp_synth(self, encoded_dim: int, n_cat_tokens: int) -> MLPSynthesizer:
        return MLPSynthesizer(
            d_in=encoded_dim,
            n_cat_tokens=n_cat_tokens,
            hidden_layers=self._mlp_layers,
            activation=self._activation,
            n_cat_emb=self._n_cat_emb,
            n_classes=None,
            embedding_learned=False,
        )

    def _create_diffuser(self) -> Diffuser:
        return Diffuser(
            device=str(self._device),
            total_steps=self._diffusion_steps,
            beta_start=self._diffusion_beta_start,
            beta_end=self._diffusion_beta_end,
            scheduler=self._scheduler,
        )

    # https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
    @torch.no_grad()  # type: ignore[untyped-decorator]
    def _generate_samples(
        self,
        mlp_synth: MLPSynthesizer,
        diffuser: Diffuser,
        artifacts: _FedTabDiffArtifacts,
        n_samples: int | None = None,
        label: Tensor | None = None,
    ) -> Tensor:

        if n_samples is None and label is None:
            raise ValueError("Either 'n_samples' or 'label' is required.")

        if label is not None:
            n_samples = len(label)
            label = label.to(self._device)

        # initialize noise
        z_norm = torch.randn((n_samples, artifacts.encoded_dim)).float()
        z_norm = z_norm.to(self._device)

        # iterate over diffusion steps
        for i in reversed(range(0, self._diffusion_steps)):
            # sample timestamps t
            t = torch.full((n_samples,), i, dtype=torch.long).to(self._device)
            # conduct forward encoder/decoder pass
            model_out = mlp_synth(z_norm, t, label)
            # reverse diffusion step, i.e. noise removal
            z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

        return z_norm

    # https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
    # noinspection PyUnnecessaryCast
    def _decode_samples(
        self, samples: Tensor, embeddings: Tensor, artifacts: _FedTabDiffArtifacts
    ) -> DataFrame:

        # split sample into numeric and categorical parts
        samples_num = samples[:, artifacts.cat_dim :]
        samples_cat = samples[:, : artifacts.cat_dim]

        # denormalize numeric attributes
        z_norm_upscaled = artifacts.num_scaler.inverse_transform(
            samples_num.cpu().numpy()
        )
        z_norm_df = DataFrame(z_norm_upscaled, columns=artifacts.num_attrs)

        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        samples_cat = samples_cat.reshape(-1, len(artifacts.cat_attrs), self._n_cat_emb)
        # Compute batch-wise distances; large embedding token counts
        # can be memory-hungry when done in a single pass.
        batch_size = 2048
        n_samples = len(samples)
        z_cat_df_list = []

        # iterate over generated categorical samples
        for i in range(0, n_samples, batch_size):
            # get batch of samples
            samples_cat_subset = samples_cat[i : i + batch_size]
            # compute pairwise distances between embeddings and generated samples
            distances = torch.cdist(x1=embeddings, x2=samples_cat_subset)
            # create temp dataframes for collection of intermediate results
            z_cat_df_temp = DataFrame(
                index=range(len(samples_cat_subset)), columns=artifacts.cat_attrs
            )

            for attr_idx, attr_name in enumerate(artifacts.cat_attrs):
                # get vocab indices for attribute
                attr_emb_idx = list(artifacts.vocab_per_attr[attr_name])
                # get distances for attribute
                attr_distances = distances[:, attr_emb_idx, attr_idx]
                # get nearest embedding index
                _, nearest_idx = torch.min(attr_distances, dim=1)
                # convert to numpy
                nearest_idx = nearest_idx.cpu().numpy()
                # map emb indices back to column indices
                z_cat_df_temp[attr_name] = np.array(attr_emb_idx)[nearest_idx]

            # collect temp DFs
            z_cat_df_list.append(z_cat_df_temp)

        # concat DFs
        z_cat_df = pd.concat(z_cat_df_list, ignore_index=True)
        # inverse transform categorical attributes
        z_cat_df = z_cat_df.apply(artifacts.label_encoder.inverse_transform)
        remove_col_prefixes(z_cat_df, artifacts.cat_attrs)
        # concat numeric and categorical attributes
        sample_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)

        return sample_decoded


def prefix_columns(df: DataFrame, cat_attrs: Iterable[str]) -> None:
    for cat_attr in cat_attrs:
        df[cat_attr] = cat_attr + "_" + df[cat_attr].astype("str")


def remove_col_prefixes(df: DataFrame, cat_attrs: Iterable[str]) -> None:
    for cat_attr in cat_attrs:
        s: Series = df[cat_attr].astype("str")
        df[cat_attr] = s.str.removeprefix(f"{cat_attr}_")
