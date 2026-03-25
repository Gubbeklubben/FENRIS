from typing import Callable, Iterable, Iterator, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbench.algorithms.fedtabdiff.diffuser import Diffuser
from fedbench.algorithms.fedtabdiff.mlpsynthesizer import MLPSynthesizer
from fedbench.core.algorithm import Synthesizer
from fedbench.core.logger import ELBOW, log_info
from fedbench.core.payload import Payload


class FedTabDiffSynthesizer(Synthesizer):
    def __init__(
        self,
        batch_size: int,
        max_batches: int,
        learning_rate: float,
        n_cat_emb: int,
        last_diff_step: int,
        diffuser_factory: Callable[[torch.device], Diffuser],
        mlp_synth_factory: Callable[[int, int], MLPSynthesizer],
    ) -> None:

        self._batch_size = batch_size
        self._max_batches = max_batches
        self._learning_rate = learning_rate
        self._n_cat_emb = n_cat_emb
        self._last_diff_step = last_diff_step
        self._diffuser_factory = diffuser_factory
        self._mlp_synth_factory = mlp_synth_factory

        self._cat_attrs: list[str] | None = None
        self._num_attrs: list[str] | None = None
        self._n_cat_tokens: int | None = None
        self._cat_dim: int | None = None
        self._encoded_dim: int | None = None
        self._vocab_per_attr: dict[str, list[int]] | None = None
        self._num_scaler: QuantileTransformer | None = None
        self._label_encoder: LabelEncoder | None = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # noinspection PyUnnecessaryCast
    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        extras = artifacts.extras["preproc-extras"]
        objects = artifacts.objects["preproc-objects"]

        self._cat_attrs = cast(list[str], extras["cat-attrs"])
        self._num_attrs = cast(list[str], extras["num-attrs"])
        self._n_cat_tokens = cast(int, extras["n-cat-tokens"])
        self._cat_dim = cast(int, extras["cat-dim"])
        self._encoded_dim = cast(int, extras["encoded-dim"])

        self._vocab_per_attr = objects["vocab-per-attr"]
        self._num_scaler = objects["num-scaler"]
        self._label_encoder = objects["label-encoder"]

    # noinspection PyUnnecessaryCast
    def train(self, request: Payload, data: DataFrame) -> Payload:
        log_info(str(self), "Start training...")

        # init loss function
        loss_fnc = nn.MSELoss()
        total_losses = []

        cat_attrs = cast(list[str], self._cat_attrs)
        num_attrs = cast(list[str], self._num_attrs)
        num_scaler = cast(QuantileTransformer, self._num_scaler)
        label_encoder = cast(LabelEncoder, self._label_encoder)

        prefix_columns(data, cat_attrs)
        cat_scaled = data[cat_attrs].apply(label_encoder.transform)
        num_scaled = num_scaler.transform(data[num_attrs].values)

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

        state_dict = request.arrays["state"]

        mlp_synth = self._mlp_synth_factory(
            cast(int, self._encoded_dim),
            cast(int, self._n_cat_tokens),
        )
        mlp_synth.load_state_dict(state_dict)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, mlp_synth.parameters()),
            lr=self._learning_rate,
        )
        # set network in training mode
        mlp_synth.train()
        mlp_synth.to(self._device)

        diffuser = self._diffuser_factory(self._device)

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
            # collect rec error losses
            total_losses.append(train_losses.detach().cpu().numpy())

            if idx + 1 >= self._max_batches:
                break

        # average of rec errors
        loss = np.mean(np.array(total_losses)).item()

        reply = Payload()
        reply.arrays["state"] = mlp_synth.state_dict()
        reply.metrics["metrics"] = {"loss": loss, "num-samples": num_samples}

        log_info(str(self), "Finished training.")
        log_info("", f"\t{ELBOW} loss: {loss}.")

        return reply

    def sample(self, request: Payload, num_rows: int, seed: int) -> DataFrame:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        log_info(str(self), "Start sampling...")

        state_dict = request.arrays["state"]
        # noinspection PyUnnecessaryCast
        mlp_synth = self._mlp_synth_factory(
            cast(int, self._encoded_dim), cast(int, self._n_cat_tokens)
        )
        mlp_synth.load_state_dict(state_dict)
        mlp_synth.to(self._device)
        diffuser = self._diffuser_factory(self._device)

        tensor = self._generate_samples(
            mlp_synth=mlp_synth,
            diffuser=diffuser,
            n_samples=num_rows,
            label=None,
        )
        log_info(str(self), "Finished sampling.")

        return self._decode_samples(
            samples=tensor, embeddings=mlp_synth.get_embeddings()
        )

    # https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
    @torch.no_grad()  # type: ignore[untyped-decorator]
    def _generate_samples(
        self,
        mlp_synth: MLPSynthesizer,
        diffuser: Diffuser,
        n_samples: int | None = None,
        label: Tensor | None = None,
    ) -> Tensor:

        if n_samples is None and label is None:
            raise ValueError("Either 'n_samples' or 'label' is required.")

        if label is not None:
            n_samples = len(label)
            label = label.to(self._device)

        # initialize noise
        z_norm = torch.randn((n_samples, self._encoded_dim)).float()
        z_norm = z_norm.to(self._device)

        # iterate over diffusion steps
        for i in reversed(range(0, self._last_diff_step)):
            # sample timestamps t
            t = torch.full((n_samples,), i, dtype=torch.long).to(self._device)
            # conduct forward encoder/decoder pass
            model_out = mlp_synth(z_norm, t, label)
            # reverse diffusion step, i.e. noise removal
            z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

        return z_norm

    # https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
    # noinspection PyUnnecessaryCast
    def _decode_samples(self, samples: Tensor, embeddings: Tensor) -> DataFrame:
        cat_attrs = cast(list[str], self._cat_attrs)
        num_attrs = cast(list[str], self._num_attrs)
        label_encoder = cast(LabelEncoder, self._label_encoder)
        num_scaler = cast(QuantileTransformer, self._num_scaler)
        vocab_per_attr = cast(dict[str, list[int]], self._vocab_per_attr)

        # split sample into numeric and categorical parts
        samples_num = samples[:, self._cat_dim :]
        samples_cat = samples[:, : self._cat_dim]

        # denormalize numeric attributes
        z_norm_upscaled = num_scaler.inverse_transform(samples_num.cpu().numpy())
        z_norm_df = DataFrame(z_norm_upscaled, columns=num_attrs)

        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        samples_cat = samples_cat.reshape(-1, len(cat_attrs), self._n_cat_emb)
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
                index=range(len(samples_cat_subset)), columns=cat_attrs
            )

            for attr_idx, attr_name in enumerate(cat_attrs):
                # get vocab indices for attribute
                attr_emb_idx = list(vocab_per_attr[attr_name])
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
        z_cat_df = z_cat_df.apply(label_encoder.inverse_transform)
        remove_col_prefixes(z_cat_df, cat_attrs)
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
