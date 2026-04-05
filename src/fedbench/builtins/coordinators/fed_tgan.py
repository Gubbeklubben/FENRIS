"""Fed-TGAN Coordinator with table similarity-aware aggregation.

Ported from the original Fed-TGAN implementation:
https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/distributed.py

Instead of uniform averaging (FedAvg), this coordinator computes table similarity
metrics for each client and uses them to weight model aggregation.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import torch
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

from fedbench.core.algorithm import SingleStepCoordinator
from fedbench.core.payload import Arrays, ArraysTarget, Payload


@dataclass(frozen=True)
class GlobalState:
    state: Arrays

    @classmethod
    def decode(cls, payload: Payload) -> Self:
        return cls(payload.arrays["state"])

    def encode(self) -> Payload:
        return Payload(arrays={"state": self.state})


@dataclass(frozen=True)
class ClientUpdate:
    state: Arrays
    count: int
    # Table similarity metrics (categorical and continuous column distributions)
    cat_distributions: dict[str, dict[str, float]]  # column_name -> {category: probability}
    num_distributions: dict[str, np.ndarray]  # column_name -> continuous values

    @classmethod
    def decode(cls, payload: Payload) -> Self:
        # noinspection PyUnnecessaryCast
        return cls(
            state=payload.arrays["state"],
            count=cast(int, payload.metrics["metrics"]["count"]),
            cat_distributions=cast(dict[str, dict[str, float]], payload.objects.get("objects", {}).get("cat_distributions", {})),
            num_distributions=cast(dict[str, np.ndarray], payload.objects.get("objects", {}).get("num_distributions", {})),
        )

    def encode(self) -> Payload:
        return Payload(
            arrays={"state": self.state},
            metrics={"metrics": {"count": self.count}},
            objects={
                "objects": {
                    "cat_distributions": self.cat_distributions,
                    "num_distributions": self.num_distributions,
                }
            },
        )


class FedTGAN(SingleStepCoordinator):
    """Fed-TGAN coordinator with table similarity-aware weighted aggregation."""

    def __init__(self) -> None:
        self._state: dict[str, torch.Tensor] | None = None
        # Store client distributions for computing global distributions
        self._client_cat_distributions: list[dict[str, dict[str, float]]] = []
        self._client_num_distributions: list[dict[str, np.ndarray]] = []
        self._client_counts: list[int] = []

    @property
    def name(self) -> str:
        return "fed_tgan"

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.TORCH

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        # noinspection PyUnnecessaryCast
        self._state = cast(dict[str, torch.Tensor], GlobalState.decode(artifacts).state)

    def configure_train(
        self, client_ids: Iterable[int]
    ) -> Iterable[tuple[int, Payload]]:

        if self._state is None:
            raise ValueError("No global state, can not configure training round.")

        for cid in client_ids:
            yield cid, GlobalState(self._state).encode()

    def aggregate_train(self, replies: Iterable[tuple[int, Payload]]) -> None:
        if not replies:
            raise ValueError("No replies, can not aggregate.")

        count: list[int] = []
        state_dicts: list[dict[str, torch.Tensor]] = []
        self._client_cat_distributions = []
        self._client_num_distributions = []
        self._client_counts = []

        for _, payload in replies:
            update = ClientUpdate.decode(payload)
            count.append(update.count)
            # noinspection PyUnnecessaryCast
            state_dicts.append(cast(dict[str, torch.Tensor], update.state))
            self._client_cat_distributions.append(update.cat_distributions)
            self._client_num_distributions.append(update.num_distributions)
            self._client_counts.append(update.count)

        total = sum(count)
        if total <= 0:
            raise ValueError(f"Total count: {count}, can not aggregate.")

        # Compute table similarity-aware weights
        weights = self._compute_aggregation_weights(
            self._client_cat_distributions,
            self._client_num_distributions,
            self._client_counts,
        )

        keys = tuple(state_dicts[0].keys())
        aggr_state: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for key in keys:
                result: torch.Tensor | None = None

                for state_dict, weight in zip(state_dicts, weights, strict=True):
                    tensor = state_dict[key].detach().cpu()
                    if result is None:
                        result = tensor * weight
                    else:
                        result = result + tensor * weight

                aggr_state[key] = result

        self._state = aggr_state

    def publish_train_artifacts(self) -> Payload:
        if self._state is None:
            raise ValueError("No global state, can not publish training artifacts.")
        return GlobalState(self._state).encode()

    def _compute_aggregation_weights(
        self,
        cat_distributions: list[dict[str, dict[str, float]]],
        num_distributions: list[dict[str, np.ndarray]],
        counts: list[int],
    ) -> list[float]:
        """Compute table similarity-aware aggregation weights.

        Ported from the original Fed-TGAN implementation (lines 767-783):
        https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/distributed.py

        Parameters
        ----------
        cat_distributions : list[dict[str, np.ndarray]]
            Categorical column distributions per client
        num_distributions : list[dict[str, np.ndarray]]
            Continuous column distributions per client
        counts : list[int]
            Number of samples per client

        Returns
        -------
        list[float]
            Aggregation weights (sums to 1)
        """
        n_clients = len(cat_distributions)

        # Compute data proportion weights
        total_count = sum(counts)
        data_weights = [float(c) / total_count for c in counts]

        # If only one client, return uniform weight
        if n_clients == 1:
            return [1.0]

        # Compute categorical similarity (Jensen-Shannon divergence)
        cat_similarities = self._compute_categorical_similarity(cat_distributions, counts)

        # Compute continuous similarity (Wasserstein distance)
        num_similarities = self._compute_continuous_similarity(num_distributions, data_weights)

        # Combine similarities
        combined_similarities = np.sum([cat_similarities, num_similarities], axis=0)

        # Convert similarity distances to weights
        # Lower distance = higher similarity = higher weight
        total_similarity = np.sum(combined_similarities)
        if total_similarity == 0:
            # All clients identical, use data proportion
            return data_weights

        similarity_weights = []
        for i in range(n_clients):
            # Invert distance: (1 - normalized_distance)
            weight = (1 - combined_similarities[i] / total_similarity) * data_weights[i]
            similarity_weights.append(weight)

        # Apply softmax for final normalization
        similarity_weights = self._softmax(np.array(similarity_weights))

        return similarity_weights.tolist()

    def _compute_categorical_similarity(
        self,
        cat_distributions: list[dict[str, dict[str, float]]],
        counts: list[int],
    ) -> np.ndarray:
        """Compute categorical column similarity using Jensen-Shannon divergence.

        Ported from lines 606-658 of the original implementation.
        """
        n_clients = len(cat_distributions)

        if not cat_distributions[0]:
            # No categorical columns
            return np.zeros(n_clients)

        # Get all categorical column names
        cat_columns = list(cat_distributions[0].keys())
        n_cat_cols = len(cat_columns)

        # Similarity matrix: (n_clients, n_cat_cols)
        similarity_matrix = np.zeros((n_clients, n_cat_cols))

        for col_idx, col_name in enumerate(cat_columns):
            # Get union of all categories across all clients
            all_categories = set()
            for client_dist in cat_distributions:
                all_categories.update(client_dist[col_name].keys())
            all_categories = sorted(all_categories)  # Sort for consistency

            # Compute global distribution (weighted by counts)
            total_count = sum(counts)
            global_dist = np.zeros(len(all_categories))

            for client_idx in range(n_clients):
                client_dict = cat_distributions[client_idx][col_name]
                weight = counts[client_idx] / total_count
                for cat_idx, category in enumerate(all_categories):
                    global_dist[cat_idx] += client_dict.get(category, 0.0) * weight

            # Compute JS divergence for each client
            for client_idx in range(n_clients):
                client_dict = cat_distributions[client_idx][col_name]
                # Convert client dict to aligned array
                client_dist = np.array([client_dict.get(cat, 0.0) for cat in all_categories])
                js_div = distance.jensenshannon(global_dist, client_dist)
                similarity_matrix[client_idx, col_idx] = js_div

        # Normalize by column (sum of distances per column)
        column_sums = np.sum(similarity_matrix, axis=0)
        for col_idx in range(n_cat_cols):
            if column_sums[col_idx] > 0:
                similarity_matrix[:, col_idx] /= column_sums[col_idx]
            else:
                # All clients identical for this column
                similarity_matrix[:, col_idx] = 1.0 / n_clients

        # Return sum across columns (total categorical similarity per client)
        return np.sum(similarity_matrix, axis=1)

    def _compute_continuous_similarity(
        self,
        num_distributions: list[dict[str, np.ndarray]],
        data_weights: list[float],
    ) -> np.ndarray:
        """Compute continuous column similarity using Wasserstein distance.

        Ported from lines 717-761 of the original implementation.
        """
        n_clients = len(num_distributions)

        if not num_distributions[0]:
            # No continuous columns
            return np.zeros(n_clients)

        # Get all continuous column names
        num_columns = list(num_distributions[0].keys())
        n_num_cols = len(num_columns)

        # Similarity matrix: (n_clients, n_num_cols)
        similarity_matrix = np.zeros((n_clients, n_num_cols))

        for col_idx, col_name in enumerate(num_columns):
            # Aggregate samples from all clients (weighted)
            aggregated_samples = []
            client_samples = []

            for client_idx in range(n_clients):
                samples = num_distributions[client_idx][col_name]
                client_samples.append(samples)
                # Weight samples by data proportion
                n_samples = int(len(samples) * data_weights[client_idx])
                if n_samples > 0:
                    # Sample with replacement to match proportion
                    resampled = np.random.choice(samples, size=n_samples, replace=True)
                    aggregated_samples.append(resampled)

            aggregated = np.concatenate(aggregated_samples)

            # Compute Wasserstein distance for each client
            for client_idx in range(n_clients):
                wd = wasserstein_distance(aggregated, client_samples[client_idx])
                similarity_matrix[client_idx, col_idx] = wd

        # Normalize by column
        column_sums = np.sum(similarity_matrix, axis=0)
        for col_idx in range(n_num_cols):
            if column_sums[col_idx] > 0:
                similarity_matrix[:, col_idx] /= column_sums[col_idx]
            else:
                # All clients identical for this column
                similarity_matrix[:, col_idx] = 1.0 / n_clients

        # Return sum across columns (total continuous similarity per client)
        return np.sum(similarity_matrix, axis=1)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax normalization (line 135-137 in original)."""
        e = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e / e.sum()
