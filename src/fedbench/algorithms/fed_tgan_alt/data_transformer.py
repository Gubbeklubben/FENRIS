"""Fed-TGAN-Alt Data Transformer.

Federated-aware data encoding using VGM (Variational Gaussian Mixture)
for continuous columns and one-hot encoding for discrete columns.

Uses sklearn.mixture.BayesianGaussianMixture directly — no dependency
on ctgan or rdt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class SpanInfo:
    """Metadata for an encoded column span.

    Attributes
    ----------
    dim
        Number of output dimensions for this column.
    activation_fn
        Activation function type: ``"tanh"`` for continuous, ``"softmax"`` for discrete.
    """

    dim: int
    activation_fn: Literal["tanh", "softmax"]


@dataclass(frozen=True)
class ColumnTransformInfo:
    """Metadata describing the transformation of one input column.

    Attributes
    ----------
    column_name
        Name of the original table column.
    column_type
        ``"continuous"`` or ``"discrete"``.
    output_info
        List of ``SpanInfo`` objects describing the encoded layout.
    output_dimensions
        Total number of output dimensions for this column.
    """

    column_name: str
    column_type: Literal["continuous", "discrete"]
    output_info: list[SpanInfo]
    output_dimensions: int


# ── Local fitting (runs on each client during Synthesizer.init) ────────── #


def fit_local_continuous(
    column_data: NDArray[Any],
    max_clusters: int = 10,
    weight_threshold: float = 0.005,
) -> dict[str, Any]:
    """Fit a BayesianGaussianMixture on a single continuous column.

    Parameters
    ----------
    column_data
        1-D array of values for one continuous column.
    max_clusters
        Upper bound on the number of Gaussian components.
    weight_threshold
        Components with weight below this value are pruned.

    Returns
    -------
    dict
        Keys ``"means"``, ``"covariances"``, ``"weights"`` — serialisable
        via ``Update.extras`` / ``Update.objects``.
    """
    if len(column_data) == 0:
        return {"means": [], "covariances": [], "weights": []}

    gm = BayesianGaussianMixture(
        n_components=min(len(column_data), max_clusters),
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.001,
        max_iter=100,
        n_init=1,
        random_state=0,
    )
    gm.fit(column_data.reshape(-1, 1))

    valid = gm.weights_ > weight_threshold
    return {
        "means": gm.means_[valid].flatten().tolist(),
        "covariances": gm.covariances_[valid].flatten().tolist(),
        "weights": gm.weights_[valid].tolist(),
    }


def fit_local_discrete(column_data: pd.Series) -> dict[str, int]:
    """Compute category frequency distribution for a single discrete column.

    Parameters
    ----------
    column_data
        Series of categorical values from one column.

    Returns
    -------
    dict
        Mapping ``{str(category): count}`` for all observed categories.
    """
    counts = column_data.value_counts()
    return {str(k): int(v) for k, v in counts.items()}


# ── Global merge (runs on server during Aggregator.aggregate_init) ─────── #


def merge_vgm_models(
    local_vgms: list[dict[str, Any]],
    client_sample_counts: list[int],
    max_clusters: int = 10,
    weight_threshold: float = 0.005,
    seed: int = 42,
    max_total_samples: int = 100_000,
) -> dict[str, Any]:
    """Merge local VGM parameters into a global VGM.

    For each client, generate synthetic samples proportional to the client's
    data size using its local VGM.  Then fit a global VGM on the combined
    synthetic samples (Fed-TGAN paper, Section 4.1, Step 1).

    Parameters
    ----------
    local_vgms
        Per-client VGM parameter dicts.
    client_sample_counts
        Number of real data rows per client.
    max_clusters
        Upper bound on the number of Gaussian components.
    weight_threshold
        Components with weight below this value are pruned.
    seed
        Random seed for sample generation.
    max_total_samples
        Hard cap on total synthetic samples generated across all clients.
        Client proportions are preserved but scaled down when the raw
        total exceeds this limit.

    Returns
    -------
    dict
        Merged VGM with keys ``"means"``, ``"covariances"``, ``"weights"``.
    """
    rng = np.random.default_rng(seed)
    all_samples: list[NDArray[Any]] = []

    # Scale down sample counts if the total would exceed the cap
    raw_total = sum(client_sample_counts)
    if raw_total > max_total_samples:
        scale = max_total_samples / raw_total
        scaled_counts = [max(1, int(n * scale)) for n in client_sample_counts]
    else:
        scaled_counts = list(client_sample_counts)

    for vgm_params, n_samples in zip(local_vgms, scaled_counts, strict=True):
        means = np.array(vgm_params["means"])
        covs = np.array(vgm_params["covariances"])
        weights = np.array(vgm_params["weights"])

        if len(means) == 0:
            continue

        # Normalize weights
        weights = weights / weights.sum()

        # Sample from each component proportionally
        component_counts = rng.multinomial(n_samples, weights)
        samples: list[NDArray[Any]] = []
        for mean, cov, count in zip(means, covs, component_counts, strict=True):
            if count > 0:
                s = rng.normal(loc=mean, scale=np.sqrt(max(cov, 1e-8)), size=count)
                samples.append(s)

        if samples:
            all_samples.append(np.concatenate(samples))

    if not all_samples:
        return {"means": [], "covariances": [], "weights": []}

    combined = np.concatenate(all_samples).reshape(-1, 1)

    gm = BayesianGaussianMixture(
        n_components=min(len(combined), max_clusters),
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.001,
        max_iter=100,
        n_init=1,
        random_state=seed,
    )
    gm.fit(combined)

    valid = gm.weights_ > weight_threshold
    return {
        "means": gm.means_[valid].flatten().tolist(),
        "covariances": gm.covariances_[valid].flatten().tolist(),
        "weights": gm.weights_[valid].tolist(),
    }


def merge_category_frequencies(
    local_freqs: list[dict[str, int]],
) -> dict[str, int]:
    """Merge category frequency distributions from all clients.

    Parameters
    ----------
    local_freqs
        Per-client frequency dicts, each mapping ``{category: count}``.

    Returns
    -------
    dict
        Global frequency dict with summed counts for all categories.
    """
    merged: dict[str, int] = {}
    for freq in local_freqs:
        for cat, count in freq.items():
            merged[cat] = merged.get(cat, 0) + count
    return merged


# ── Global Data Transformer ────────────────────────────────────────────── #


class GlobalDataTransformer:
    """Encodes/decodes tabular data using global VGMs and one-hot encoding.

    Built from merged VGM parameters (continuous) and merged category sets
    (discrete). Mirrors the encoding scheme from CTGAN but with no
    dependency on external ``ctgan`` or ``rdt`` libraries, enabling
    stateless serialization across federated clients.
    """

    def __init__(self) -> None:
        self._column_transform_info_list: list[ColumnTransformInfo] = []
        self._continuous_info: dict[str, dict[str, Any]] = {}
        self._discrete_info: dict[str, OneHotEncoder] = {}
        self._column_order: list[str] = []
        self.output_info_list: list[list[SpanInfo]] = []
        self.output_dimensions: int = 0

    def fit_global(
        self,
        column_order: list[str],
        column_types: dict[str, Literal["continuous", "discrete"]],
        global_vgms: dict[str, dict[str, Any]],
        global_categories: dict[str, list[str]],
    ) -> None:
        """Build the global transformer from merged federated parameters.

        Parameters
        ----------
        column_order
            Names of all columns in the desired output order.
        column_types
            Mapping ``{column_name: "continuous" | "discrete"}``.
        global_vgms
            Merged VGM parameters per continuous column, keyed by name.
        global_categories
            Sorted category lists per discrete column, keyed by name.
        """
        self._column_order = column_order
        self._column_transform_info_list = []
        self.output_info_list = []
        self.output_dimensions = 0

        for col_name in column_order:
            col_type = column_types[col_name]

            if col_type == "continuous":
                vgm = global_vgms[col_name]
                n_components = len(vgm["means"])
                if n_components == 0:
                    n_components = 1  # Fallback: single component

                output_dim = 1 + n_components
                info = ColumnTransformInfo(
                    column_name=col_name,
                    column_type="continuous",
                    output_info=[
                        SpanInfo(1, "tanh"),
                        SpanInfo(n_components, "softmax"),
                    ],
                    output_dimensions=output_dim,
                )
                self._continuous_info[col_name] = vgm

            else:  # discrete
                cats = global_categories[col_name]
                ohe = OneHotEncoder(
                    categories=[cats],
                    sparse_output=False,
                    handle_unknown="ignore",
                )
                # Fit on the known categories
                ohe.fit(np.array(cats).reshape(-1, 1))
                self._discrete_info[col_name] = ohe

                n_cats = len(cats)
                output_dim = n_cats
                info = ColumnTransformInfo(
                    column_name=col_name,
                    column_type="discrete",
                    output_info=[SpanInfo(n_cats, "softmax")],
                    output_dimensions=output_dim,
                )

            self._column_transform_info_list.append(info)
            self.output_info_list.append(info.output_info)
            self.output_dimensions += output_dim

    def to_dict(self) -> dict[str, Any]:
        """Serialise the transformer to a plain-Python dict.

        The returned dict is JSON-safe (only ``str``, ``int``, ``float``
        and ``list`` values) and can be transmitted through
        ``Update.extras`` or any other serialisation layer.

        Returns
        -------
        dict
            Keys ``"column_order"``, ``"column_types"``,
            ``"global_vgms"``, ``"global_categories"``.
        """
        column_types: dict[str, str] = {}
        global_categories: dict[str, list[str]] = {}
        for info in self._column_transform_info_list:
            column_types[info.column_name] = info.column_type
            if info.column_type == "discrete":
                ohe = self._discrete_info[info.column_name]
                global_categories[info.column_name] = list(ohe.categories_[0])

        return {
            "column_order": list(self._column_order),
            "column_types": column_types,
            "global_vgms": {
                col: dict(vgm) for col, vgm in self._continuous_info.items()
            },
            "global_categories": global_categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalDataTransformer":
        """Reconstruct a transformer from a dict produced by ``to_dict()``.

        Parameters
        ----------
        data
            Dict with keys ``"column_order"``, ``"column_types"``,
            ``"global_vgms"``, ``"global_categories"``.

        Returns
        -------
        GlobalDataTransformer
            Fully initialised transformer ready for ``transform`` /
            ``inverse_transform`` calls.
        """
        transformer = cls()
        col_types_raw: dict[str, str] = data["column_types"]
        col_types: dict[str, Literal["continuous", "discrete"]] = {}
        for k, v in col_types_raw.items():
            if v not in ("continuous", "discrete"):
                raise ValueError(f"Unknown column type '{v}' for column '{k}'.")
            col_types[k] = v  # type: ignore[assignment]
        transformer.fit_global(
            column_order=data["column_order"],
            column_types=col_types,
            global_vgms=data.get("global_vgms", {}),
            global_categories=data.get("global_categories", {}),
        )
        return transformer

    def transform(self, df: pd.DataFrame) -> NDArray[Any]:
        """Encode a DataFrame into a numeric matrix.

        Parameters
        ----------
        df
            Input data with the same columns used during ``fit_global``.

        Returns
        -------
        ndarray
            Float32 matrix of shape ``(n_rows, output_dimensions)``.
        """
        column_data_list: list[NDArray[Any]] = []

        for info in self._column_transform_info_list:
            col = info.column_name
            values = df[col].to_numpy()

            if info.column_type == "continuous":
                column_data_list.append(self._transform_continuous(col, values))
            else:
                column_data_list.append(self._transform_discrete(col, values))

        return np.concatenate(column_data_list, axis=1).astype(np.float32)

    def inverse_transform(self, data: NDArray[Any]) -> pd.DataFrame:
        """Decode a numeric matrix back into a DataFrame.

        Parameters
        ----------
        data
            Encoded matrix of shape ``(n_rows, output_dimensions)`` as
            produced by ``transform``.

        Returns
        -------
        DataFrame
            Reconstructed table with the original column order.
        """
        recovered: dict[str, NDArray[Any]] = {}
        st = 0

        for info in self._column_transform_info_list:
            dim = info.output_dimensions
            col_data = data[:, st : st + dim]

            if info.column_type == "continuous":
                recovered[info.column_name] = self._inverse_continuous(
                    info.column_name, col_data
                )
            else:
                recovered[info.column_name] = self._inverse_discrete(
                    info.column_name, col_data
                )
            st += dim

        return pd.DataFrame(recovered)

    # ── Continuous transform ───────────────────────────────────────────── #

    def _transform_continuous(
        self,
        col: str,
        values: NDArray[Any],
    ) -> NDArray[Any]:
        """VGM mode-specific normalization → [normalized_value, mode_one_hot]."""
        vgm = self._continuous_info[col]
        means = np.array(vgm["means"])
        stds = np.sqrt(np.maximum(np.array(vgm["covariances"]), 1e-8))
        weights = np.array(vgm["weights"])
        n_components = len(means)

        if n_components == 0:
            # Fallback: no valid components, output zeros
            return np.zeros((len(values), 2), dtype=np.float32)

        values = values.astype(np.float64).reshape(-1, 1)

        # Compute probability of each value under each component
        # p(x | k) ∝ w_k * N(x; μ_k, σ_k)
        log_probs = np.zeros((len(values), n_components))
        for k in range(n_components):
            log_probs[:, k] = (
                np.log(weights[k] + 1e-12)
                - 0.5 * ((values[:, 0] - means[k]) / stds[k]) ** 2
                - np.log(stds[k] + 1e-12)
            )

        # Select best component
        component_idx = np.argmax(log_probs, axis=1)

        # Normalize within selected component: (value - μ_k) / (4 * σ_k)
        selected_means = means[component_idx]
        selected_stds = stds[component_idx]
        normalized = (values[:, 0] - selected_means) / (4.0 * selected_stds)
        normalized = np.clip(normalized, -0.99, 0.99)

        # One-hot encode the component
        one_hot = np.zeros((len(values), n_components), dtype=np.float32)
        one_hot[np.arange(len(values)), component_idx] = 1.0

        return np.column_stack([normalized, one_hot]).astype(np.float32)

    def _inverse_continuous(
        self,
        col: str,
        col_data: NDArray[Any],
    ) -> NDArray[Any]:
        """Reverse VGM normalization."""
        vgm = self._continuous_info[col]
        means = np.array(vgm["means"])
        stds = np.sqrt(np.maximum(np.array(vgm["covariances"]), 1e-8))

        if len(means) == 0:
            return np.zeros(len(col_data))

        normalized = col_data[:, 0]
        component_probs = col_data[:, 1:]
        component_idx = np.argmax(component_probs, axis=1)

        selected_means = means[component_idx]
        selected_stds = stds[component_idx]
        values: NDArray[Any] = normalized * 4.0 * selected_stds + selected_means

        return values

    # ── Discrete transform ─────────────────────────────────────────────── #

    def _transform_discrete(
        self,
        col: str,
        values: NDArray[Any],
    ) -> NDArray[Any]:
        """One-hot encode discrete values."""
        ohe = self._discrete_info[col]
        # Categories were stored as strings (fit_local_discrete uses str(k)),
        # so cast incoming values to str to match and avoid np.isnan on objects.
        str_values = np.array([str(v) for v in values]).reshape(-1, 1)
        result: NDArray[Any] = ohe.transform(str_values).astype(np.float32)
        return result

    def _inverse_discrete(
        self,
        col: str,
        col_data: NDArray[Any],
    ) -> NDArray[Any]:
        """Reverse one-hot encoding."""
        ohe = self._discrete_info[col]
        indices = np.argmax(col_data, axis=1)
        categories = ohe.categories_[0]
        result: NDArray[Any] = categories[indices]
        return result

    # ── Helper for conditional vector / column ID mapping ──────────────── #

    def convert_column_name_value_to_id(
        self,
        column_name: str,
        value: str,
    ) -> dict[str, int]:
        """Get the IDs of a column/value pair for conditional generation.

        Parameters
        ----------
        column_name
            Name of a discrete column in the fitted transformer.
        value
            Category value to look up.

        Returns
        -------
        dict
            Keys ``"discrete_column_id"``, ``"column_id"``, ``"value_id"``.

        Raises
        ------
        ValueError
            If *column_name* is not found or *value* is unknown.
        """
        discrete_counter = 0
        column_id = 0
        for info in self._column_transform_info_list:
            if info.column_name == column_name:
                break
            if info.column_type == "discrete":
                discrete_counter += 1
            column_id += 1
        else:
            raise ValueError(f"Column '{column_name}' not found.")

        ohe = self._discrete_info[column_name]
        encoded = ohe.transform(np.array([[value]]))
        if encoded.sum() == 0:
            raise ValueError(f"Value '{value}' not found in column '{column_name}'.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": int(np.argmax(encoded)),
        }
