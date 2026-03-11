"""Fed-TGAN-Alt: Alternative Fed-TGAN implementation.

Based on: Zhao et al., "Fed-TGAN: Federated Learning Framework for
Synthesizing Tabular Data" (2021). arXiv:2108.07927

Architecture follows the FedBench Algorithm / Coordinator / Synthesizer
ABC pattern. The CTGAN-style encoding (VGM for continuous, one-hot for
discrete) and table-similarity-aware weighting are implemented in
separate modules under this package.
"""

from collections.abc import Iterable
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor, nn, optim

from fedbench.core.algorithm import (
    Algorithm,
    Coordinator,
    SingleStepCoordinator,
    Synthesizer,
)
from fedbench.core.data import TableSchema
from fedbench.core.logger import (
    ELBOW,
    TEE,
    log_debug,
    log_info,
    log_warning,
)
from fedbench.core.update import Objects, Update

from .data_sampler import DataSampler
from .data_transformer import (
    GlobalDataTransformer,
    SpanInfo,
    fit_local_continuous,
    fit_local_discrete,
    merge_category_frequencies,
    merge_vgm_models,
)
from .discriminator import Discriminator
from .generator import Generator
from .weighting import compute_client_weights


def _split_cat_num(schema: TableSchema) -> tuple[list[str], list[str]]:
    """Split column names by kind into categorical and continuous lists.

    Parameters
    ----------
    schema
        Table schema defining column names and their semantic kinds.

    Returns
    -------
    tuple
        Two lists: (categorical_column_names, continuous_column_names).
        Categorical includes both ``categorical`` and ``binary`` kinds.
        Continuous includes both ``continuous`` and ``integer`` kinds.
    """
    cat_attrs = [c.name for c in schema.columns if c.kind in ("categorical", "binary")]
    num_attrs = [c.name for c in schema.columns if c.kind in ("continuous", "integer")]
    return cat_attrs, num_attrs


# ── Activation / loss helpers ──────────────────────────────────────────── #


def _gumbel_softmax(
    logits: Tensor,
    tau: float = 0.2,
    hard: bool = False,
    eps: float = 1e-10,
) -> Tensor:
    """Gumbel-Softmax with logit clamping for numerical stability.

    Parameters
    ----------
    logits
        Unnormalized log-probabilities ``(batch_size, num_categories)``.
    tau
        Temperature parameter. Lower values produce harder samples.
    hard
        If ``True``, use the straight-through estimator.
    eps
        Small constant passed to ``torch.nn.functional.gumbel_softmax``.

    Returns
    -------
    Tensor
        Soft (or hard) one-hot samples with the same shape as *logits*.

    Raises
    ------
    ValueError
        If ``gumbel_softmax`` keeps returning NaN after 10 retries.
    """
    logits = logits.clamp(-20.0, 20.0)
    for _ in range(10):
        transformed = torch.nn.functional.gumbel_softmax(
            logits, tau=tau, hard=hard, eps=eps, dim=-1
        )
        if not torch.isnan(transformed).any():
            return transformed
    raise ValueError("gumbel_softmax returning NaN.")


def _apply_activate(
    data: Tensor,
    output_info: list[list[SpanInfo]],
) -> Tensor:
    """Apply column-specific activation functions to generator output.

    Reconstructs tabular structure by applying ``tanh`` to continuous
    components and Gumbel-Softmax to discrete components, according to
    the encoding layout defined in *output_info*.

    Parameters
    ----------
    data
        Raw generator output (batch_size, total_data_dim).
    output_info
        Per-column encoding metadata with activation function specs.

    Returns
    -------
    Tensor
        Activated data with the same shape as input.
    """
    data_t: list[Tensor] = []
    st = 0
    for col_info in output_info:
        for span_info in col_info:
            ed = st + span_info.dim
            if span_info.activation_fn == "tanh":
                data_t.append(torch.tanh(data[:, st:ed]))
            elif span_info.activation_fn == "softmax":
                data_t.append(_gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)


def _cond_loss(
    fake: Tensor,
    cond: Tensor,
    mask: Tensor,
    output_info: list[list[SpanInfo]],
) -> Tensor:
    """Compute cross-entropy loss for conditioned discrete column generation.

    Enforces consistency between the generated samples and their
    target conditional column assignments during training.

    Parameters
    ----------
    fake
        Generator output (batch_size, data_dim).
    cond
        One-hot encoded conditional vector (batch_size, cond_dim).
    mask
        Binary mask indicating which rows are conditioned (batch_size, n_discrete).
    output_info
        Encoding layout describing data and conditional dimensions.

    Returns
    -------
    Tensor
        Scalar cross-entropy loss, masked over batches.
    """
    loss: list[Tensor] = []
    st = 0
    st_c = 0
    for col_info in output_info:
        for span_info in col_info:
            if len(col_info) != 1 or span_info.activation_fn != "softmax":
                st += span_info.dim
            else:
                ed = st + span_info.dim
                ed_c = st_c + span_info.dim
                tmp = nn.functional.cross_entropy(
                    fake[:, st:ed],
                    torch.argmax(cond[:, st_c:ed_c], dim=1),
                    reduction="none",
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c
    if not loss:
        return torch.tensor(0.0, device=fake.device)
    loss_stacked = torch.stack(loss, dim=1)
    return (loss_stacked * mask).sum() / fake.size(0)


# ── Algorithm factory ──────────────────────────────────────────────────── #


class FedTGANAlt(Algorithm):
    """Alternative Fed-TGAN implementation (Zhao et al., 2021).

    Horizontal federated GAN using CTGAN-style VGM encoding,
    conditional sampling, and table-similarity-aware client weighting.

    Parameters
    ----------
    embedding_dim
        Size of the noise vector concatenated with the conditional vector.
    generator_dim
        Hidden layer sizes for each Residual block in the generator.
        Defaults to ``[256, 256]``.
    discriminator_dim
        Hidden layer sizes in the discriminator.
        Defaults to ``[256, 256]``.
    generator_lr
        Adam learning rate for the generator.
    discriminator_lr
        Adam learning rate for the discriminator.
    batch_size
        Training batch size (must be even and divisible by *pac*).
    max_batches
        Maximum number of gradient steps per communication round.
    discriminator_steps
        Number of discriminator updates per generator update.
    pac
        Number of samples packed together for PacGAN.
    max_clusters
        Upper bound on VGM mixture components per continuous column.
    weight_threshold
        VGM components with weight below this value are pruned.
    max_total_samples
        Hard cap on total synthetic samples generated when merging
        local VGM models on the server.  Client proportions are
        preserved but scaled down when the raw total exceeds this
        limit.  Defaults to ``100_000``.
    log_frequency
        If ``True``, use log-frequency weighting for discrete columns.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: list[int] | None = None,
        discriminator_dim: list[int] | None = None,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        batch_size: int = 500,
        max_batches: int = 10,
        discriminator_steps: int = 1,
        pac: int = 10,
        max_clusters: int = 10,
        weight_threshold: float = 0.005,
        max_total_samples: int = 100_000,
        log_frequency: bool = True,
    ) -> None:
        if generator_dim is None:
            generator_dim = [256, 256]
        if discriminator_dim is None:
            discriminator_dim = [256, 256]

        if embedding_dim < 1:
            raise ValueError("Expecting embedding_dim >= 1.")
        if generator_lr <= 0 or generator_lr > 0.1:
            raise ValueError("Expecting 0 < generator_lr <= 0.1.")
        if discriminator_lr <= 0 or discriminator_lr > 0.1:
            raise ValueError("Expecting 0 < discriminator_lr <= 0.1.")
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("Expecting even batch_size >= 2.")
        if pac < 1:
            raise ValueError("Expecting pac >= 1.")
        if batch_size % pac != 0:
            raise ValueError("Expecting batch_size divisible by pac.")
        if discriminator_steps < 1:
            raise ValueError("Expecting discriminator_steps >= 1.")
        if max_batches < 1:
            raise ValueError("Expecting max_batches >= 1.")
        if max_clusters < 1:
            raise ValueError("Expecting max_clusters >= 1.")
        if max_total_samples < 1:
            raise ValueError("Expecting max_total_samples >= 1.")

        self._cfg: dict[str, Any] = {
            "embedding-dim": embedding_dim,
            "generator-dim": generator_dim,
            "discriminator-dim": discriminator_dim,
            "generator-lr": generator_lr,
            "discriminator-lr": discriminator_lr,
            "batch-size": batch_size,
            "max-batches": max_batches,
            "discriminator-steps": discriminator_steps,
            "pac": pac,
            "max-clusters": max_clusters,
            "weight-threshold": weight_threshold,
            "max-total-samples": max_total_samples,
            "log-frequency": log_frequency,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

    def create_coordinator(self) -> Coordinator:
        return FedTGANAltCoordinator(self._cfg)

    def create_synthesizer(self) -> Synthesizer:
        return FedTGANAltSynthesizer(self._cfg)


# ── Coordinator (server-side) ──────────────────────────────────────────── #


class FedTGANAltCoordinator(SingleStepCoordinator):
    """Server-side coordinator for Fed-TGAN-Alt.

    Merges local VGM statistics into a global data transformer,
    computes similarity-aware client weights, and performs
    weighted FedAvg on generator and discriminator state dicts.

    Parameters
    ----------
    cfg
        Algorithm hyperparameter dictionary produced by ``FedTGANAlt``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._cat_attrs: list[str] = []
        self._num_attrs: list[str] = []
        self._int_attrs: list[str] = []
        self._binary_attrs: list[str] = []
        self._schema_column_order: list[str] = []
        self._transformer: GlobalDataTransformer | None = None
        self._client_weights: list[float] = []
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._category_probs: np.ndarray = np.array([])
        self._g_state: dict[str, Tensor] | None = None
        self._d_state: dict[str, Tensor] | None = None

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"generator": "torch", "discriminator": "torch"}

    @property
    def global_state(self) -> Update | None:
        return self._create_update()

    def configure_fed_init(
        self,
        seed: int,
        schema: TableSchema,
        client_ids: Iterable[int],
    ) -> Iterable[tuple[int, Update]]:
        """Broadcast hyperparameters and seed to all clients.

        Seeds the global RNG and yields one configuration ``Update`` per
        client containing ``max-clusters`` and ``weight-threshold`` so
        that each client can fit local VGM models consistently.

        Parameters
        ----------
        seed
            Global random seed broadcast to all clients.
        schema
            Table schema used to split columns into categorical/continuous.
        client_ids
            IDs of all participating clients.

        Yields
        ------
        tuple of (int, Update)
            ``(client_id, config_update)`` pairs.
        """
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self._cat_attrs, self._num_attrs = _split_cat_num(schema)
        self._int_attrs = [c.name for c in schema.columns if c.kind == "integer"]
        self._schema_column_order = [c.name for c in schema.columns]
        self._binary_attrs = [c.name for c in schema.columns if c.kind == "binary"]

        for cid in client_ids:
            update = Update()
            update.extras["config"] = {
                "max-clusters": self._cfg["max-clusters"],
                "weight-threshold": self._cfg["weight-threshold"],
            }
            yield cid, update

    def aggregate_fed_init(self, replies: Iterable[tuple[int, Update]]) -> None:
        """Aggregate client init replies to build the global model.

        Merges per-client VGM parameters and category frequencies into a
        global ``GlobalDataTransformer``, computes table-similarity-aware
        client weights, and initialises empty Generator/Discriminator
        state dicts ready for the first training round.

        Parameters
        ----------
        replies
            ``(client_id, init_reply)`` pairs.  Each reply must have
            ``"cat-freqs"`` and ``"cont-vgms"`` in ``.objects`` and
            ``"init-extras"`` (containing ``"num-samples"``) in ``.extras``.
        """
        # Collect local statistics from all clients
        all_cat_freqs: list[dict[str, dict[str, int]]] = []
        all_cont_vgms: list[dict[str, dict[str, Any]]] = []
        client_sample_counts: list[int] = []

        for _, reply in replies:
            extras = reply.extras["init-extras"]
            all_cat_freqs.append(
                cast(dict[str, dict[str, int]], reply.objects["cat-freqs"])
            )
            all_cont_vgms.append(
                cast(dict[str, dict[str, Any]], reply.objects["cont-vgms"])
            )
            client_sample_counts.append(cast(int, extras["num-samples"]))

        # Merge category frequencies → global categories + probabilities
        global_categories: dict[str, list[str]] = {}
        category_probs_per_col: list[np.ndarray] = []
        for col in self._cat_attrs:
            col_freqs = [cf.get(col, {}) for cf in all_cat_freqs]
            merged = merge_category_frequencies(col_freqs)
            sorted_cats = sorted(merged.keys())
            global_categories[col] = sorted_cats
            # Compute frequency-based probabilities (matching DataSampler)
            counts = np.array([merged[c] for c in sorted_cats], dtype=np.float64)
            if self._cfg["log-frequency"]:
                counts = np.log(counts + 1)
            probs = counts / counts.sum()
            category_probs_per_col.append(probs)

        # Flat probability vector over all categories (for sampling)
        if category_probs_per_col:
            self._category_probs = np.concatenate(category_probs_per_col)
            self._category_probs = self._category_probs / self._category_probs.sum()
        else:
            self._category_probs = np.array([], dtype=np.float64)

        # Merge VGM models → global VGMs
        global_vgms: dict[str, dict[str, Any]] = {}
        for col in self._num_attrs:
            col_vgms = [
                cv.get(col, {"means": [], "covariances": [], "weights": []})
                for cv in all_cont_vgms
            ]
            global_vgms[col] = merge_vgm_models(
                col_vgms,
                client_sample_counts,
                max_clusters=self._cfg["max-clusters"],
                weight_threshold=self._cfg["weight-threshold"],
                max_total_samples=self._cfg["max-total-samples"],
            )

        # Compute table-similarity-aware client weights
        self._client_weights = compute_client_weights(
            cat_freqs=all_cat_freqs,
            cont_vgms=all_cont_vgms,
            client_sample_counts=client_sample_counts,
            cat_columns=self._cat_attrs,
            cont_columns=self._num_attrs,
        )
        log_info(str(self), "Computed client weights:")
        for i, w in enumerate(self._client_weights):
            log_info("", f"\t{TEE} client {i}: {w:.4f}")

        # Build global transformer
        column_order = self._cat_attrs + self._num_attrs
        column_types: dict[str, Literal["continuous", "discrete"]] = {
            c: "discrete" for c in self._cat_attrs
        }
        column_types.update({c: "continuous" for c in self._num_attrs})

        self._transformer = GlobalDataTransformer()
        self._transformer.fit_global(
            column_order=column_order,
            column_types=column_types,
            global_vgms=global_vgms,
            global_categories=global_categories,
        )

        data_dim = self._transformer.output_dimensions
        cond_dim = self._transformer.cond_dim

        # Initialize Generator and Discriminator
        embedding_dim = self._cfg["embedding-dim"]
        generator = Generator(
            embedding_dim + cond_dim,
            self._cfg["generator-dim"],
            data_dim,
        )
        discriminator = Discriminator(
            data_dim + cond_dim,
            self._cfg["discriminator-dim"],
            pac=self._cfg["pac"],
        )

        self._g_state = generator.state_dict()
        self._d_state = discriminator.state_dict()

        log_info(str(self), f"Initialized G ({data_dim}d) + D, cond_dim={cond_dim}")

    def aggregate_train(self, replies: Iterable[tuple[int, Update]]) -> None:
        """Aggregate client train replies via weighted FedAvg.

        Performs a weighted average of generator and discriminator state
        dicts using the similarity-aware client weights computed during
        ``aggregate_fed_init``.

        Parameters
        ----------
        replies
            ``(client_id, train_reply)`` pairs.  Each reply must have
            ``"generator"`` and ``"discriminator"`` in ``.arrays``.

        Raises
        ------
        ValueError
            If *replies* is empty.
        """
        replies_list = list(replies)
        if not replies_list:
            raise ValueError("No replies, cannot aggregate.")

        g_state_dicts: list[dict[str, Tensor]] = []
        d_state_dicts: list[dict[str, Tensor]] = []

        for _, reply in replies_list:
            g_state_dicts.append(cast(dict[str, Tensor], reply.arrays["generator"]))
            d_state_dicts.append(cast(dict[str, Tensor], reply.arrays["discriminator"]))

        # Use precomputed similarity weights
        weights = self._client_weights
        if len(weights) != len(g_state_dicts):
            log_warning(
                str(self),
                f"Client weight count ({len(weights)}) does not match "
                f"reply count ({len(g_state_dicts)}); falling back to "
                f"uniform weights.",
            )
            weights = [1.0 / len(g_state_dicts)] * len(g_state_dicts)

        self._g_state = _weighted_average_state_dicts(g_state_dicts, weights)
        self._d_state = _weighted_average_state_dicts(d_state_dicts, weights)

    def _create_update(self) -> Update:
        assert self._transformer is not None
        return Update(
            arrays={
                "generator": cast(dict[str, Tensor], self._g_state),
                "discriminator": cast(dict[str, Tensor], self._d_state),
            },
            objects={
                "transformer": self._transformer.to_dict(),
                "weights": {"weights": self._client_weights},
            },
            extras={
                "model-info": {
                    "embedding-dim": self._cfg["embedding-dim"],
                    "batch-size": self._cfg["batch-size"],
                    "pac": self._cfg["pac"],
                    "log-frequency": self._cfg["log-frequency"],
                    "category-probs": self._category_probs.tolist(),
                    "integer-columns": self._int_attrs,
                    "binary-columns": self._binary_attrs,
                    "schema-column-order": self._schema_column_order,
                },
            },
        )


# ── Synthesizer (client-side) ──────────────────────────────────────────── #


class FedTGANAltSynthesizer(Synthesizer):
    """Client-side synthesizer for Fed-TGAN-Alt.

    Fits local VGM / discrete statistics during init, runs WGAN-GP
    training with CTGAN-style conditional sampling during train,
    and generates synthetic rows during sample.

    Parameters
    ----------
    cfg
        Algorithm hyperparameter dictionary produced by ``FedTGANAlt``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._device = cfg["device"]
        self._rng: np.random.Generator = np.random.default_rng(0)

    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"generator": "torch", "discriminator": "torch"}

    def fed_init(
        self,
        request: Update,
        seed: int,
        schema: TableSchema,
        data: DataFrame,
    ) -> Update:
        """Fit local statistics and return them to the coordinator.

        Computes per-column VGM parameters for continuous columns and
        category frequency distributions for discrete columns from the
        client's local data.

        Parameters
        ----------
        request
            Configuration update from ``configure_fed_init`` containing
            ``"max-clusters"`` and ``"weight-threshold"`` in
            ``.extras["config"]``.
        seed
            Client-local random seed (reserved for future use).
        schema
            Table schema identifying categorical/continuous column kinds.
        data
            Client's local training data.

        Returns
        -------
        Update
            Reply with ``"cat-freqs"`` and ``"cont-vgms"`` in ``.objects``
            and ``"init-extras"`` (``{"num-samples": int}``) in ``.extras``.
        """
        cat_attrs, num_attrs = _split_cat_num(schema)
        extras_cfg = request.extras["config"]
        max_clusters = cast(int, extras_cfg["max-clusters"])
        weight_threshold = cast(float, extras_cfg["weight-threshold"])

        # Fit local VGM for each continuous column
        cont_vgms: dict[str, dict[str, Any]] = {}
        for col in num_attrs:
            col_data = data[col].to_numpy()
            cont_vgms[col] = fit_local_continuous(
                col_data,
                max_clusters=max_clusters,
                weight_threshold=weight_threshold,
            )

        # Compute category frequencies for each discrete column
        cat_freqs: dict[str, dict[str, int]] = {}
        for col in cat_attrs:
            cat_freqs[col] = fit_local_discrete(data[col])

        num_samples = len(data)

        log_debug(
            str(self),
            f"Init: {num_samples} samples, "
            f"{len(cont_vgms)} continuous, "
            f"{len(cat_freqs)} discrete columns",
        )

        reply = Update()
        reply.objects["cat-freqs"] = cat_freqs
        reply.objects["cont-vgms"] = cont_vgms
        reply.extras["init-extras"] = {"num-samples": num_samples}
        return reply

    def train(self, request: Update, data: DataFrame) -> Update:
        """Run one local WGAN-GP training round.

        Loads the global Generator and Discriminator state dicts, trains
        them on the client's local data using CTGAN-style conditional
        sampling, and returns the updated state dicts.

        Parameters
        ----------
        request
            Global state from ``FedTGANAltCoordinator.global_state``
            containing the transformer, model state dicts, and training
            hyperparameters.
        data
            Client's local training data.

        Returns
        -------
        Update
            Reply with ``"generator"`` and ``"discriminator"`` state dicts
            in ``.arrays`` and training metrics in ``.metrics["metrics"]``.
        """
        log_info(str(self), "Start training...")

        # Extract global state
        transformer = GlobalDataTransformer.from_dict(
            request.objects["transformer"]
        )
        output_info = transformer.output_info_list
        data_dim = transformer.output_dimensions

        g_state = cast(dict[str, Tensor], request.arrays["generator"])
        d_state = cast(dict[str, Tensor], request.arrays["discriminator"])

        model_info = request.extras["model-info"]
        embedding_dim = cast(int, model_info["embedding-dim"])
        batch_size = cast(int, model_info["batch-size"])
        pac = cast(int, model_info["pac"])
        log_frequency = cast(bool, model_info["log-frequency"])

        # Transform local data
        train_data = transformer.transform(data)

        # Build local data sampler
        sampler = DataSampler(train_data, output_info, log_frequency)
        cond_dim = sampler.dim_cond_vec()

        # Reconstruct networks
        generator = Generator(
            embedding_dim + cond_dim,
            self._cfg["generator-dim"],
            data_dim,
        ).to(self._device)
        generator.load_state_dict(g_state)

        discriminator = Discriminator(
            data_dim + cond_dim,
            self._cfg["discriminator-dim"],
            pac=pac,
        ).to(self._device)
        discriminator.load_state_dict(d_state)

        optimizer_g = optim.Adam(
            generator.parameters(),
            lr=self._cfg["generator-lr"],
            betas=(0.5, 0.9),
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=self._cfg["discriminator-lr"],
            betas=(0.5, 0.9),
        )

        mean = torch.zeros(batch_size, embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // batch_size, 1)
        generator.train()
        discriminator.train()

        total_loss_g = 0.0
        total_loss_d = 0.0
        num_steps = 0

        for step_idx in range(min(steps_per_epoch, self._cfg["max-batches"])):
            # ── Discriminator step(s) ──
            for _ in range(self._cfg["discriminator-steps"]):
                fakez = torch.normal(mean=mean, std=std)
                condvec = sampler.sample_condvec(batch_size)

                if condvec is None:
                    c1, col, opt = None, None, None
                    real = sampler.sample_data(train_data, batch_size, col, opt)
                else:
                    c1, _m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                    perm = np.arange(batch_size)
                    self._rng.shuffle(perm)
                    real = sampler.sample_data(
                        train_data, batch_size, col[perm], opt[perm]
                    )
                    c2 = c1[perm]

                fake = generator(fakez)
                fakeact = _apply_activate(fake, output_info).detach()

                real_t = torch.from_numpy(real.astype("float32")).to(self._device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real_t, c2], dim=1)
                else:
                    fake_cat = fakeact
                    real_cat = real_t

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                pen = discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device
                )
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                optimizer_d.zero_grad(set_to_none=False)
                (loss_d + pen).backward()
                optimizer_d.step()

            # ── Generator step ──
            fakez = torch.normal(mean=mean, std=std)
            condvec = sampler.sample_condvec(batch_size)

            if condvec is None:
                c1, m1 = None, None
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = generator(fakez)
            fakeact = _apply_activate(fake, output_info)

            if c1 is not None:
                y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = discriminator(fakeact)

            if condvec is None:
                cross_entropy = torch.tensor(0.0, device=self._device)
            else:
                cross_entropy = _cond_loss(fake, c1, m1, output_info)

            loss_g = -torch.mean(y_fake) + cross_entropy

            optimizer_g.zero_grad(set_to_none=False)
            loss_g.backward()
            optimizer_g.step()

            total_loss_g += loss_g.detach().cpu().item()
            total_loss_d += loss_d.detach().cpu().item()
            num_steps += 1

        avg_loss_g = total_loss_g / max(num_steps, 1)
        avg_loss_d = total_loss_d / max(num_steps, 1)
        num_samples = len(train_data)

        log_info(str(self), "Finished training.")
        log_info("", f"\t{TEE} loss_g: {avg_loss_g:.4f}")
        log_info("", f"\t{ELBOW} loss_d: {avg_loss_d:.4f}")

        reply = Update()
        reply.arrays["generator"] = generator.state_dict()
        reply.arrays["discriminator"] = discriminator.state_dict()
        reply.metrics["metrics"] = {
            "loss-g": avg_loss_g,
            "loss-d": avg_loss_d,
            "num-samples": num_samples,
        }
        return reply

    def sample(
        self,
        request: Update,
        num_rows: int,
        seed: int,
    ) -> DataFrame:
        """Generate synthetic tabular data from the trained generator.

        Parameters
        ----------
        request
            Global state from ``FedTGANAltCoordinator.global_state``
            containing the transformer and generator state dict.
        num_rows
            Number of synthetic rows to generate.
        seed
            Random seed for reproducible sampling.

        Returns
        -------
        DataFrame
            Synthetic data with the same columns as the training data.
        """
        log_info(str(self), "Start sampling...")

        transformer = GlobalDataTransformer.from_dict(
            request.objects["transformer"]
        )
        output_info = transformer.output_info_list
        data_dim = transformer.output_dimensions

        g_state = cast(dict[str, Tensor], request.arrays["generator"])
        model_info = request.extras["model-info"]
        embedding_dim = cast(int, model_info["embedding-dim"])
        batch_size = cast(int, model_info["batch-size"])

        # We need cond_dim to reconstruct generator input size
        cond_dim = transformer.cond_dim

        generator = Generator(
            embedding_dim + cond_dim,
            self._cfg["generator-dim"],
            data_dim,
        ).to(self._device)
        generator.load_state_dict(g_state)
        generator.eval()

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Recover category probabilities for data-aware sampling
        raw_probs = model_info.get("category-probs")
        category_probs = np.array(raw_probs, dtype=np.float64) if raw_probs else None

        data_list: list[np.ndarray] = []
        steps = num_rows // batch_size + 1

        for _ in range(steps):
            mean = torch.zeros(batch_size, embedding_dim, device=self._device)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            if cond_dim > 0:
                condvec = _sample_condvec_from_info(
                    output_info, batch_size, category_probs, rng=rng
                )
                c1 = torch.from_numpy(condvec).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            with torch.no_grad():
                fake = generator(fakez)
                fakeact = _apply_activate(fake, output_info)

            data_list.append(fakeact.detach().cpu().numpy())

        data_concat = np.concatenate(data_list, axis=0)[:num_rows]
        result = transformer.inverse_transform(data_concat)

        # Round integer columns back to int
        int_cols = cast(list[str], model_info.get("integer-columns", []))
        for col in int_cols:
            if col in result.columns:
                result[col] = result[col].round().astype(int)

        # Restore binary columns to int (inverse_transform returns strings)
        binary_cols = cast(list[str], model_info.get("binary-columns", []))
        for col in binary_cols:
            if col in result.columns:
                result[col] = (
                    pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)
                )

        # Restore original schema column order
        schema_order = cast(
            list[str] | None, model_info.get("schema-column-order")
        )
        if schema_order:
            result = result[[c for c in schema_order if c in result.columns]]

        log_info(str(self), f"Sampled {len(result)} rows.")
        return result


# ── Utilities ──────────────────────────────────────────────────────────── #


def _weighted_average_state_dicts(
    state_dicts: list[dict[str, Tensor]],
    weights: list[float],
) -> dict[str, Tensor]:
    """Compute weighted average of multiple PyTorch state dicts.

    Parameters
    ----------
    state_dicts
        List of ``state_dict`` mappings (one per client).
    weights
        Corresponding aggregation weights (must sum to 1).

    Returns
    -------
    dict
        Merged state dict with the same keys as the inputs.
    """
    keys = tuple(state_dicts[0].keys())
    result: dict[str, Tensor] = {}

    with torch.no_grad():
        for key in keys:
            acc: Tensor | None = None
            for sd, w in zip(state_dicts, weights, strict=True):
                tensor = sd[key].detach().cpu()
                if acc is None:
                    acc = tensor * w
                else:
                    acc = acc + tensor * w
            assert acc is not None  # guaranteed by non-empty state_dicts
            result[key] = acc

    return result


def _sample_condvec_from_info(
    output_info: list[list[SpanInfo]],
    batch_size: int,
    category_probs: np.ndarray | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a one-hot conditional vector for generation.

    When *category_probs* is provided, categories are sampled
    proportionally — mirroring CTGAN's ``sample_original_condvec``.
    Otherwise falls back to uniform sampling.

    Parameters
    ----------
    output_info
        Per-column list of ``SpanInfo`` describing the encoded layout.
    batch_size
        Number of conditional vectors to produce.
    category_probs
        Flat probability vector over all categories (length must equal
        the total number of one-hot positions).  ``None`` triggers
        uniform fallback.

    Returns
    -------
    ndarray of shape ``(batch_size, n_categories)``
        One-hot encoded conditional vectors.
    """
    n_categories = 0
    for col_info in output_info:
        if len(col_info) == 1 and col_info[0].activation_fn == "softmax":
            n_categories += col_info[0].dim

    if n_categories == 0:
        return np.zeros((batch_size, 0), dtype=np.float32)

    cond = np.zeros((batch_size, n_categories), dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()

    if category_probs is not None and len(category_probs) == n_categories:
        # Data-aware sampling: pick category indices proportional to frequency
        probs = category_probs.copy()
        probs = probs / probs.sum()  # re-normalise for safety
        chosen = rng.choice(n_categories, size=batch_size, p=probs)
        cond[np.arange(batch_size), chosen] = 1.0
    else:
        # Fallback: uniform over discrete columns × categories
        discrete_spans: list[tuple[int, int]] = []
        cond_st = 0
        for col_info in output_info:
            if len(col_info) == 1 and col_info[0].activation_fn == "softmax":
                discrete_spans.append((cond_st, col_info[0].dim))
                cond_st += col_info[0].dim
        for i in range(batch_size):
            col_idx = rng.integers(len(discrete_spans))
            st, dim = discrete_spans[col_idx]
            cat_idx = rng.integers(dim)
            cond[i, st + cat_idx] = 1.0

    return cond
