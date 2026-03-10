"""Fed-TGAN-Alt Generator network.

Uses residual blocks following CTGAN's architecture.
Each Residual block concatenates its output with the input,
progressively growing the hidden dimension.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class Residual(nn.Module):  # type: ignore[misc]
    """Residual block: Linear → BatchNorm → ReLU, concatenated with input."""

    def __init__(self, i: int, o: int) -> None:
        super().__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_: Tensor) -> Tensor:
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):  # type: ignore[misc]
    """Generator for Fed-TGAN-Alt / CTGAN.

    Generates synthetic data by stacking residual blocks, where each block
    concatenates its output with the input, linearly growing the hidden
    dimension. Final layer projects to the encoded data dimension.

    Parameters
    ----------
    embedding_dim
        Size of input (noise + conditional vector concatenated).
    generator_dim
        Sequence of hidden layer sizes for each Residual block.
    data_dim
        Output dimension (total encoded data size).
    """

    def __init__(
        self,
        embedding_dim: int,
        generator_dim: Sequence[int],
        data_dim: int,
    ) -> None:
        super().__init__()
        dim = embedding_dim
        seq: list[nn.Module] = []
        for item in generator_dim:
            seq.append(Residual(dim, item))
            dim += item  # Residual concatenates output with input
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_: Tensor) -> Tensor:
        """Generate synthetic data from noise + conditional vector.

        Parameters
        ----------
        input_
            Concatenated noise and conditional vector (batch_size, embedding_dim).

        Returns
        -------
        Tensor
            Generated data (batch_size, data_dim).
        """
        return self.seq(input_)
