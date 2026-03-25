import functools
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from fedbench.core.algorithm import (
    Algorithm,
    ComponentSpec,
    Coordinator,
    GlobalInitArtifacts,
    Synthesizer,
    coordinator_spec,
    synthesizer_spec,
)
from fedbench.core.data import TableSchema
from fedbench.core.payload import Payload

# Relative imports for algorithm specifics.
from .coordinator import FedTabDiffCoordinator
from .diffuser import Diffuser
from .mlpsynthesizer import MLPSynthesizer
from .synthesizer import FedTabDiffSynthesizer, prefix_columns


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

        self._n_cat_emb = n_cat_emb

        mlp_synth_factory: Callable[[int, int], MLPSynthesizer] = functools.partial(
            MLPSynthesizer,
            hidden_layers=mlp_layers,
            activation=activation,
            n_cat_emb=n_cat_emb,
            n_classes=None,
            embedding_learned=False,
        )
        self._mlp_synth_factory = mlp_synth_factory

        self._synth_factory: Callable[[], Synthesizer] = functools.partial(
            FedTabDiffSynthesizer,
            batch_size=batch_size,
            max_batches=max_batches,
            n_cat_emb=n_cat_emb,
            learning_rate=learning_rate,
            last_diff_step=diffusion_steps,
            diffuser_factory=functools.partial(
                Diffuser,
                total_steps=diffusion_steps,
                beta_start=diffusion_beta_start,
                beta_end=diffusion_beta_end,
                scheduler=scheduler,
            ),
            mlp_synth_factory=mlp_synth_factory,
        )

    @property
    def coordinator_spec(self) -> ComponentSpec[Coordinator]:
        return coordinator_spec(
            FedTabDiffCoordinator, {"initial-state": "torch", "state": "torch"}
        )

    @property
    def synthesizer_spec(self) -> ComponentSpec[Synthesizer]:
        return synthesizer_spec(self._synth_factory, {"state": "torch"})

    def global_init(
        self, seed: int, schema: TableSchema, dataset: DataFrame
    ) -> GlobalInitArtifacts | None:

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        cat_attrs, num_attrs = split_cat_num(schema)
        prefix_columns(dataset, cat_attrs)

        num_scaler = QuantileTransformer(
            n_quantiles=len(dataset),
            output_distribution="normal",
            random_state=seed,
        )
        num_scaler.fit(dataset[num_attrs].values)

        vocab_classes = sorted(np.unique(dataset[cat_attrs]))
        label_encoder = LabelEncoder().fit(vocab_classes)
        cat_scaled = dataset[cat_attrs].apply(label_encoder.transform)

        vocab_per_attr = {attr: set(cat_scaled[attr]) for attr in cat_attrs}
        n_cat_tokens = len(vocab_classes)
        cat_dim = self._n_cat_emb * len(cat_attrs)
        encoded_dim = cat_dim + len(num_attrs)

        synth_artifacts = Payload()

        synth_artifacts.objects["preproc-objects"] = {
            "num-scaler": num_scaler,
            "label-encoder": label_encoder,
            "vocab-per-attr": vocab_per_attr,
        }
        synth_artifacts.extras["preproc-extras"] = {
            "cat-attrs": cat_attrs,
            "num-attrs": num_attrs,
            "n-cat-tokens": n_cat_tokens,
            "cat-dim": cat_dim,
            "encoded-dim": encoded_dim,
        }
        mlp_synth = self._mlp_synth_factory(encoded_dim, n_cat_tokens)

        return GlobalInitArtifacts(
            coordinator=Payload(arrays={"initial-state": mlp_synth.state_dict()}),
            synthesizer=synth_artifacts,
        )
