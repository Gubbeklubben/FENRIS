"""Bayesian Gaussian Mixture Transformer for Fed-TGAN.

Ported from the original Fed-TGAN implementation:
https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/features/transformers.py

Citation:
    Zhao, Zilong, Robert Birke, Aditya Kunar, and Lydia Y. Chen.
    "Fed-TGAN: Federated Learning Framework for Synthesizing Tabular Data."
    arXiv preprint arXiv:2108.07927 (2021).

This transformer uses Bayesian Gaussian Mixture Models to encode continuous
columns with mode-specific normalization, which better preserves multi-modal
distributions compared to simple scaling.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture


class BGMTransformer:
    """Model continuous columns with Bayesian GMM and normalize them.

    For each continuous column:
    1. Fit a Bayesian Gaussian Mixture with n_clusters modes
    2. Select active modes (weight > eps threshold)
    3. For each value:
       - Normalize per mode: (x - μ_i) / (4σ_i)
       - Sample mode from probability distribution
       - Output: [normalized_value, one_hot_mode_selection]

    Categorical columns are one-hot encoded.

    Attributes:
        n_clusters: Maximum number of Gaussian modes per column
        eps: Weight threshold for mode selection (default: 0.005)
        meta: Column metadata (type, ranges, etc.)
        model: List of fitted BGM models per column
        components: Active components mask per column
        output_info: List of (dimension, activation_type) per output
        output_dim: Total output dimensionality
    """

    def __init__(self, n_clusters: int = 10, eps: float = 0.005):
        """Initialize BGM transformer.

        Parameters
        ----------
        n_clusters : int
            Upper bound on number of Gaussian modes (default: 10)
        eps : float
            Weight threshold for keeping a mode (default: 0.005)
        """
        self.n_clusters = n_clusters
        self.eps = eps
        self.meta: list[dict[str, Any]] = []
        self.model: list[BayesianGaussianMixture | None] = []
        self.components: list[np.ndarray | None] = []
        self.output_info: list[tuple[int, str]] = []
        self.output_dim = 0

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: list[str],
        continuous_columns: list[str],
    ) -> None:
        """Fit BGM models to continuous columns.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
        categorical_columns : list[str]
            Names of categorical columns
        continuous_columns : list[str]
            Names of continuous columns
        """
        self.meta = []
        self.model = []
        self.components = []
        self.output_info = []
        self.output_dim = 0

        # Build metadata
        for col in data.columns:
            if col in categorical_columns:
                unique_vals = data[col].value_counts().index.tolist()
                self.meta.append(
                    {
                        "name": col,
                        "type": "categorical",
                        "size": len(unique_vals),
                        "i2s": unique_vals,  # index to string mapping
                    }
                )
            elif col in continuous_columns:
                self.meta.append(
                    {
                        "name": col,
                        "type": "continuous",
                        "min": float(data[col].min()),
                        "max": float(data[col].max()),
                    }
                )
            else:
                raise ValueError(f"Column {col} not in categorical or continuous lists")

        # Fit models
        data_array = data.values
        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                # Fit Bayesian GMM
                gm = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    n_init=1,
                    max_iter=1000,  # Avoid sklearn convergence warnings
                    random_state=42,
                )
                gm.fit(data_array[:, id_].reshape(-1, 1))
                self.model.append(gm)

                # Select active components (weight > eps)
                comp = gm.weights_ > self.eps
                self.components.append(comp)

                # Output: 1 normalized value + n_active_modes one-hot
                self.output_info.append((1, "tanh"))
                self.output_info.append((int(np.sum(comp)), "softmax"))
                self.output_dim += 1 + int(np.sum(comp))

            else:  # categorical
                self.model.append(None)
                self.components.append(None)
                self.output_info.append((info["size"], "softmax"))
                self.output_dim += info["size"]

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted BGM models.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform

        Returns
        -------
        np.ndarray
            Transformed data: [normalized_values, mode_one_hots, category_one_hots]
        """
        data_array = data.values
        values = []

        for id_, info in enumerate(self.meta):
            current = data_array[:, id_]

            if info["type"] == "continuous":
                current = current.reshape(-1, 1)

                # Get means and stds for all modes
                model = self.model[id_]
                assert model is not None  # Guaranteed by fit() for continuous columns
                means = model.means_.reshape(1, self.n_clusters)
                stds = np.sqrt(model.covariances_).reshape(1, self.n_clusters)

                # Normalize per mode: (x - μ) / (4σ)
                features = (current - means) / (4 * stds)

                # Get mode probabilities
                probs = model.predict_proba(current)
                components = self.components[id_]
                assert (
                    components is not None
                )  # Guaranteed by fit() for continuous columns
                n_opts = int(np.sum(components))

                # Keep only active modes
                features = features[:, components]
                probs = probs[:, components]

                # Sample mode for each value
                opt_sel = np.zeros(len(data), dtype=int)
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / pp.sum()
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                # Extract normalized value for selected mode
                idx = np.arange(len(features))
                features_selected = features[idx, opt_sel].reshape(-1, 1)
                features_selected = np.clip(features_selected, -0.99, 0.99)

                # Create one-hot for selected mode
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1

                values.append(features_selected)
                values.append(probs_onehot)

            else:  # categorical
                # One-hot encode
                col_t = np.zeros([len(data), info["size"]])
                idx = np.array([info["i2s"].index(val) for val in current])
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1).astype(np.float32)

    def inverse_transform(
        self, data: np.ndarray, sigmas: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Inverse transform from encoded representation back to original scale.

        Parameters
        ----------
        data : np.ndarray
            Transformed data to decode
        sigmas : np.ndarray | None
            Optional noise to add during decoding (for sampling)

        Returns
        -------
        pd.DataFrame
            Decoded data in original scale
        """
        columns: dict[str, Any] = {}
        st = 0

        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                # Extract normalized value and mode one-hot
                u = data[:, st]
                components = self.components[id_]
                assert (
                    components is not None
                )  # Guaranteed by fit() for continuous columns
                n_modes = int(np.sum(components))
                v = data[:, st + 1 : st + 1 + n_modes]

                # Optionally add noise for sampling diversity
                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)

                # Expand mode selection to full n_clusters
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, components] = v
                v = v_t

                st += 1 + n_modes

                # Get selected mode parameters
                model = self.model[id_]
                assert model is not None  # Guaranteed by fit() for continuous columns
                means = model.means_.reshape(-1)
                stds = np.sqrt(model.covariances_).reshape(-1)
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]

                # Denormalize: x = u * 4σ + μ
                columns[info["name"]] = u * 4 * std_t + mean_t

            else:  # categorical
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                # Store as list to preserve the original dtype
                columns[info["name"]] = [info["i2s"][i] for i in idx]

        return pd.DataFrame(columns)

    def get_output_info(self) -> list[tuple[int, str]]:
        """Get output information for each encoded column.

        Returns
        -------
        list[tuple[int, str]]
            List of (dimension, activation_type) tuples
            activation_type is 'tanh' for normalized values, 'softmax' for one-hots
        """
        return self.output_info.copy()

    def get_output_dim(self) -> int:
        """Get total output dimensionality.

        Returns
        -------
        int
            Total dimension after encoding all columns
        """
        return self.output_dim
