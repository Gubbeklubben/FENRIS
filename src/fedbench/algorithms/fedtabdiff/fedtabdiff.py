from collections.abc import Iterable
from typing import Any, Iterator, Literal, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbench.core.algorithm import (
    Algorithm,
    Coordinator,
    SingleStepCoordinator,
    Synthesizer,
)
from fedbench.core.data import TableSchema
from fedbench.core.logger import ELBOW, TEE, log_debug, log_info, log_warning
from fedbench.core.update import Extras, Objects, Update

# Relative imports for algorithm specifics.
from .diffuser import Diffuser
from .mlpsynth import MLPSynthesizer


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


def prefix_columns(df: DataFrame, cat_attrs: Iterable[str]) -> None:
    for cat_attr in cat_attrs:
        df[cat_attr] = cat_attr + "_" + df[cat_attr].astype("str")


def remove_col_prefixes(df: DataFrame, cat_attrs: Iterable[str]) -> None:
    for cat_attr in cat_attrs:
        s: Series = df[cat_attr].astype("str")
        df[cat_attr] = s.str.removeprefix(f"{cat_attr}_")


# https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
def init_model(cfg: dict[str, Any]) -> tuple[MLPSynthesizer, Diffuser]:
    synthesizer = MLPSynthesizer(
        d_in=cfg["encoded-dim"],
        hidden_layers=cfg["mlp-layers"],
        activation=cfg["activation"],
        n_cat_tokens=cfg["n-cat-tokens"],
        n_cat_emb=cfg["n-cat-emb"],
        n_classes=None,
        embedding_learned=False,
    )
    diffuser = Diffuser(
        total_steps=cfg["diffusion-steps"],
        beta_start=cfg["diffusion-beta-start"],
        beta_end=cfg["diffusion-beta-end"],
        device=cfg["device"],
        scheduler=cfg["scheduler"],
    )
    return synthesizer, diffuser


# https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
@torch.no_grad()  # type: ignore[untyped-decorator]
def generate_samples(
    synthesizer: MLPSynthesizer,
    diffuser: Diffuser,
    encoded_dim: int,
    last_diff_step: int,
    n_samples: int | None = None,
    label: Tensor | None = None,
) -> Tensor:

    if n_samples is None and label is None:
        raise ValueError("Either 'n_samples' or 'label' is required.")

    device = next(synthesizer.parameters()).device

    if label is not None:
        n_samples = len(label)
        label = label.to(device)

    # initialize noise
    z_norm = torch.randn((n_samples, encoded_dim)).float()
    z_norm = z_norm.to(device)

    # iterate over diffusion steps
    for i in reversed(range(0, last_diff_step)):
        # sample timestamps t
        t = torch.full((n_samples,), i, dtype=torch.long).to(device)

        # conduct forward encoder/decoder pass
        model_out = synthesizer(z_norm, t, label)

        # reverse diffusion step, i.e. noise removal
        z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

    return z_norm


# https://github.com/sattarov/FedTabDiff/blob/main/fedtabdiff_modules.py
def decode_samples(
    samples: Tensor,
    cat_dim: int,
    n_cat_emb: int,
    num_attrs: list[str],
    cat_attrs: list[str],
    num_scaler: QuantileTransformer,
    vocab_per_attr: dict[str, Iterable[int]],
    label_encoder: LabelEncoder,
    embeddings: Tensor,
) -> DataFrame:

    # split sample into numeric and categorical parts
    samples_num = samples[:, cat_dim:]
    samples_cat = samples[:, :cat_dim]

    # denormalize numeric attributes
    z_norm_upscaled = num_scaler.inverse_transform(samples_num.cpu().numpy())
    z_norm_df = DataFrame(z_norm_upscaled, columns=num_attrs)

    # reshape back to batch_size * n_dim_cat * cat_emb_dim
    samples_cat = samples_cat.reshape(-1, len(cat_attrs), n_cat_emb)

    # Compute batch-wise distances; large embedding token counts can be memory costly
    # when done in a single pass.
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


class FedTabDiff(Algorithm):
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
    ):

        # Validation apparently not done by mlp_synth/diffuser components.
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

        self._cfg = {
            "batch-size": batch_size,
            "max-batches": max_batches,
            "n-cat-emb": n_cat_emb,
            "learning-rate": learning_rate,
            "mlp-layers": mlp_layers,
            "activation": activation,
            "diffusion-steps": diffusion_steps,
            "diffusion-beta-start": diffusion_beta_start,
            "diffusion-beta-end": diffusion_beta_end,
            "scheduler": scheduler,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

    def create_coordinator(self) -> Coordinator:
        return FedTabDiffCoordinator(self._cfg)

    def create_synthesizer(self) -> Synthesizer:
        return FedTabDiffSynthesizer(self._cfg)


class FedTabDiffCoordinator(SingleStepCoordinator):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg: dict[str, Any] = cfg
        self._cat_attrs: list[str] | None = None
        self._num_attrs: list[str] | None = None
        self._client_preproc_objects: Objects | None = None
        self._client_preproc_extras: Extras | None = None
        self._state: dict[str, Tensor] | None = None

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"arrays": "torch"}

    @property
    def global_state(self) -> Update | None:
        return self._create_update()

    def configure_fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Iterable[tuple[int, Update]]:

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self._cat_attrs, self._num_attrs = split_cat_num(schema)

        initialize_qt: bool = True
        # TODO! Framework must somehow guarantee reproducible mapping to
        #  partition_id if this is to be reliable.
        for cid in client_ids:
            update = Update()
            cfg: Extras = {"initialize-qt": initialize_qt}
            update.extras["config"] = cfg
            initialize_qt = False  # Only one client
            yield cid, update

    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        vocab_classes: set[str] = set()
        num_scaler = None

        for _, reply in replies:
            if "preproc-objects" in reply.objects:
                preproc_obj = reply.objects["preproc-objects"]
                num_scaler = preproc_obj["num-scaler"]

            extras = reply.extras["preproc-extras"]
            # noinspection PyUnnecessaryCast
            client_vocab_classes: list[str] = cast(list[str], extras["vocab-classes"])
            vocab_classes = vocab_classes | set(client_vocab_classes)

        vocab_classes_srt: list[str] = sorted(vocab_classes)
        label_encoder = LabelEncoder().fit(vocab_classes_srt)

        vocab_per_attr = {}
        # noinspection PyUnnecessaryCast
        for col in cast(list[str], self._cat_attrs):
            prefix = f"{col}_"
            tokens = tuple(t for t in vocab_classes_srt if t.startswith(prefix))
            if tokens:
                ids = label_encoder.transform(tokens)
                vocab_per_attr[col] = set(ids)
            else:
                vocab_per_attr[col] = set()

        self._client_preproc_objects = {
            "num-scaler": num_scaler,
            "label-encoder": label_encoder,
            "vocab-per-attr": vocab_per_attr,
        }
        # noinspection PyUnnecessaryCast
        cat_attrs = cast(list[str], self._cat_attrs)
        # noinspection PyUnnecessaryCast
        num_attrs = cast(list[str], self._num_attrs)
        extras = {
            "cat-attrs": cat_attrs,
            "num-attrs": num_attrs,
            "n-cat-tokens": len(vocab_classes),
            "cat-dim": self._cfg["n-cat-emb"] * len(cat_attrs),
            "encoded-dim": self._cfg["n-cat-emb"] * len(cat_attrs) + len(num_attrs),
        }
        self._client_preproc_extras = extras

        preproc = self._client_preproc_objects | self._client_preproc_extras
        mlp_synth, _ = init_model(self._cfg | preproc)
        self._state = mlp_synth.state_dict()

    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        if not replies:
            raise ValueError("No replies, can not aggregate.")

        num_samples: list[int] = []
        state_dicts: list[dict[str, Tensor]] = []

        for _, reply in replies:
            # noinspection PyUnnecessaryCast
            num_samples.append(cast(int, reply.metrics["metrics"]["num-samples"]))
            # noinspection PyUnnecessaryCast
            state_dicts.append(cast(dict[str, Tensor], reply.arrays["arrays"]))

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
        # noinspection PyUnnecessaryCast
        return Update(
            arrays={"arrays": cast(dict[str, Tensor], self._state)},
            objects={"preproc-objects": cast(Objects, self._client_preproc_objects)},
            extras={"preproc-extras": cast(Extras, self._client_preproc_extras)},
        )


class FedTabDiffSynthesizer(Synthesizer):
    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"arrays": "torch"}

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._max_batches = cfg["max-batches"]
        self._device = cfg["device"]

    def fed_init(
        self,
        request: Update,
        seed: int,
        schema: TableSchema,
        data: DataFrame,
    ) -> Update:

        cat_attrs, num_attrs = split_cat_num(schema)
        prefix_columns(data, cat_attrs)

        # Aggregator has no data access, pick the simplest possible route
        # and let one client perform the task performed centrally in
        # https://github.com/sattarov/FedTabDiff/blob/main/main.py
        cfg: Extras = request.extras["config"]
        # noinspection PyUnnecessaryCast
        initialize_qt: bool = cast(bool, cfg["initialize-qt"])

        if initialize_qt:
            num_scaler = QuantileTransformer(
                n_quantiles=len(data),
                output_distribution="normal",
                random_state=seed,
            )
            num_scaler.fit(data[num_attrs].values)
        else:
            num_scaler = None

        vocab_classes = list(np.unique(data[cat_attrs]))
        log_debug(str(self), "")
        log_debug("", f"\t{TEE} vocab_classes: {vocab_classes}")
        log_debug("", f"\t{TEE} type: {type(vocab_classes)}")
        try:
            log_debug("", f"\t{ELBOW} type[, ...]: {type(vocab_classes[0])}")
        except IndexError:
            pass

        update = Update()
        update.extras["preproc-extras"] = {"vocab-classes": vocab_classes}
        if num_scaler is not None:
            # Pickle it for now
            update.objects["preproc-objects"] = {"num-scaler": num_scaler}
        return update

    def train(self, request: Update, data: DataFrame) -> Update:
        log_info(str(self), "Start training...")
        arrays = request.arrays["arrays"]
        preproc = request.objects["preproc-objects"] | request.extras["preproc-extras"]

        mlp_synth, diffuser = init_model(self._cfg | preproc)
        mlp_synth.load_state_dict(arrays)
        # set_parameters(mlp_synth, arrays)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, mlp_synth.parameters()),
            lr=self._cfg["learning-rate"],
        )

        # init loss function
        loss_fnc = nn.MSELoss()
        total_losses = []

        cat_attrs = preproc["cat-attrs"]
        prefix_columns(data, cat_attrs)
        label_encoder = preproc["label-encoder"]
        cat_scaled = data[cat_attrs].apply(label_encoder.transform)

        num_attrs = preproc["num-attrs"]
        num_scaler = preproc["num-scaler"]
        num_scaled = num_scaler.transform(data[num_attrs].values)

        tensor_dataset = TensorDataset(
            torch.tensor(cat_scaled.values, dtype=torch.long),
            torch.tensor(num_scaled, dtype=torch.float),
        )
        torch_loader = DataLoader(
            tensor_dataset, batch_size=self._cfg["batch-size"], shuffle=True
        )

        # Adapt unsupervised fedbench training to orig alg loop
        # Tmp solution.
        def loader() -> Iterator[tuple[Tensor, Tensor, Tensor | None]]:
            for cat, num in torch_loader:
                yield cat, num, None

        # set network in training mode
        mlp_synth.train()
        mlp_synth.to(self._device)

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

            if idx >= self._max_batches:
                break

        # average of rec errors
        loss = np.mean(np.array(total_losses)).item()
        reply = Update()
        reply.arrays["arrays"] = mlp_synth.state_dict()
        reply.metrics["metrics"] = {"loss": loss, "num-samples": num_samples}
        log_info(str(self), "Finished training.")
        log_info("", f"\t{ELBOW} loss: {loss}.")
        return reply

    def sample(self, request: Update, num_rows: int, seed: int) -> DataFrame:
        log_info(str(self), "Start sampling...")
        arrays = request.arrays["arrays"]
        preproc = request.objects["preproc-objects"] | request.extras["preproc-extras"]

        mlp_synth, diffuser = init_model(self._cfg | preproc)
        mlp_synth.load_state_dict(arrays)
        mlp_synth.to(self._device)

        tensor = generate_samples(
            mlp_synth,
            diffuser,
            encoded_dim=preproc["encoded-dim"],
            last_diff_step=self._cfg["diffusion-steps"],
            n_samples=num_rows,
            label=None,
        )
        log_info(str(self), "Finished sampling.")
        return decode_samples(
            tensor,
            cat_dim=preproc["cat-dim"],
            n_cat_emb=self._cfg["n-cat-emb"],
            num_attrs=preproc["num-attrs"],
            cat_attrs=preproc["cat-attrs"],
            num_scaler=preproc["num-scaler"],
            vocab_per_attr=preproc["vocab-per-attr"],
            label_encoder=preproc["label-encoder"],
            embeddings=mlp_synth.get_embeddings(),
        )
