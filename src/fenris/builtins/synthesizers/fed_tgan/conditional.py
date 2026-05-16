"""Conditional vector system for Fed-TGAN training-by-sampling.

Ported from the original Fed-TGAN implementation:
https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/synthesizers/ctgan.py

This implements training-by-sampling where the generator is conditioned on
specific categorical values to ensure all categories are learned, preventing
mode collapse.
"""

import numpy as np
import torch
import torch.nn.functional as F


def maximum_interval(output_info: list[tuple[int, str]]) -> int:
    """Find maximum dimension among all output columns.

    Parameters
    ----------
    output_info : list[tuple[int, str]]
        List of (dimension, activation_type) per column

    Returns
    -------
    int
        Maximum dimension
    """
    max_interval = 0
    for dim, _ in output_info:
        max_interval = max(max_interval, dim)
    return max_interval


def random_choice_prob_index(
    probs: np.ndarray, rng: np.random.Generator, axis: int = 1
) -> np.ndarray:
    """Sample indices from probability distributions.

    For each row, samples an index according to the probability distribution.

    Parameters
    ----------
    probs : np.ndarray
        Probability matrix (rows are distributions)
    rng : np.random.Generator
        Local random number generator (for reproducibility)
    axis : int
        Axis along which to sample (default: 1)

    Returns
    -------
    np.ndarray
        Sampled indices for each row
    """
    r = np.expand_dims(rng.random(probs.shape[1 - axis]), axis=axis)
    result: np.ndarray = (probs.cumsum(axis=axis) > r).argmax(axis=axis)
    return result


class Cond:
    """Conditional vector generator for training-by-sampling.

    During training, randomly selects a categorical column and a specific
    category within that column to condition the generator on. This ensures
    the generator learns all categories, especially rare ones.

    Attributes
    ----------
        model: Category values per categorical column
        interval: Start position and size for each categorical column
        n_col: Number of categorical columns
        n_opt: Total one-hot dimension across all categorical columns
        p: Probability matrix for sampling categories
    """

    def __init__(self, data: np.ndarray, output_info: list[tuple[int, str]]):
        """Initialize conditional vector generator.

        Parameters
        ----------
        data : np.ndarray
            Transformed training data (after BGM encoding)
        output_info : list[tuple[int, str]]
            Output info from transformer: (dimension, activation_type)
        """
        self.model = []
        st = 0
        counter = 0

        # Extract categories from each softmax column
        for dim, activation in output_info:
            if activation == "tanh":
                # Skip continuous/normalized columns
                st += dim
                continue
            elif activation == "softmax":
                # Extract argmax for this categorical column
                ed = st + dim
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                raise ValueError(f"Unknown activation: {activation}")

        assert st == data.shape[1], "Output info doesn't match data dimensions"

        # Build probability matrix for sampling
        interval_list: list[tuple[int, int]] = []
        self.n_col = 0
        self.n_opt = 0
        st = 0

        # Initialize probability matrix
        max_dim = maximum_interval(output_info)
        self.p = np.zeros((counter, max_dim))

        for dim, activation in output_info:
            if activation == "tanh":
                st += dim
                continue
            elif activation == "softmax":
                ed = st + dim

                # Compute log-weighted frequency for each category
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)  # Log frequency
                tmp = tmp / np.sum(tmp)  # Normalize

                self.p[self.n_col, :dim] = tmp
                interval_list.append((self.n_opt, dim))
                self.n_opt += dim
                self.n_col += 1
                st = ed

        self.interval = np.asarray(interval_list)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample conditional vectors for a training batch.

        Randomly selects which column to condition on and which category
        within that column for each sample in the batch.

        Parameters
        ----------
        batch_size : int
            Number of samples
        rng : np.random.Generator
            Local random number generator (for reproducibility)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
            - vec: Conditional vector (one-hot, size: batch x n_opt)
            - mask: Column mask (one-hot, size: batch x n_col)
            - idx: Selected column indices (size: batch)
            - opt: Selected category indices within columns (size: batch)
            Returns None if no categorical columns exist
        """
        if self.n_col == 0:
            return None

        # Randomly select which column to condition on for each sample
        idx = rng.choice(np.arange(self.n_col), batch_size)

        # Initialize conditional vector and mask
        vec = np.zeros((batch_size, self.n_opt), dtype=np.float32)
        mask = np.zeros((batch_size, self.n_col), dtype=np.float32)
        mask[np.arange(batch_size), idx] = 1  # Mark selected column

        # Sample category within selected column according to frequency
        opt = random_choice_prob_index(self.p[idx], rng)

        # Set one-hot in conditional vector
        for i in range(batch_size):
            vec[i, self.interval[idx[i], 0] + opt[i]] = 1

        return vec, mask, idx, opt

    def sample_original_training_data_prob(
        self, batch_size: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        """Sample conditional vectors matching training data distribution.

        Used during sampling/generation to produce categories with frequencies
        similar to the training data.

        Parameters
        ----------
        batch_size : int
            Number of samples
        rng : np.random.Generator
            Local random number generator (for reproducibility)

        Returns
        -------
        np.ndarray | None
            Conditional vector (size: batch x n_opt)
            Returns None if no categorical columns exist
        """
        if self.n_col == 0:
            return None

        vec = np.zeros((batch_size, self.n_opt), dtype=np.float32)

        # Randomly select column
        idx = rng.choice(np.arange(self.n_col), batch_size)

        for i in range(batch_size):
            col = idx[i]
            # Pick a random category from observed training data for this column
            pick = int(rng.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1

        return vec


def cond_loss(
    data: torch.Tensor,
    output_info: list[tuple[int, str]],
    c: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """Compute conditional loss to enforce generator respects conditional vector.

    Computes cross-entropy between generator's output for the conditioned column
    and the target category from the conditional vector.

    Parameters
    ----------
    data : torch.Tensor
        Generated data (before final activation)
    output_info : list[tuple[int, str]]
        Output info: (dimension, activation_type)
    c : torch.Tensor
        Conditional vector (one-hot)
    m : torch.Tensor
        Mask indicating which column was conditioned

    Returns
    -------
    torch.Tensor
        Conditional cross-entropy loss
    """
    loss = []
    st = 0
    st_c = 0

    for dim, activation in output_info:
        if activation == "tanh":
            st += dim
        elif activation == "softmax":
            ed = st + dim
            ed_c = st_c + dim

            # Cross-entropy between generated and conditional target
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction="none",
            )
            loss.append(tmp)

            st = ed
            st_c = ed_c

    loss_stacked = torch.stack(loss, dim=1)  # (batch, n_categorical_cols)

    # Apply mask and average
    return (loss_stacked * m).sum() / data.size(0)
