"""Fed-TGAN Generator network.

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
    """Generator for Fed-TGAN / CTGAN.

    Parameters
    ----------
    embedding_dim
        Size of the random noise vector + conditional vector.
    generator_dim
        Sequence of hidden layer sizes for Residual blocks.
    data_dim
        Output dimension (encoded data dimension).
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
        return self.seq(input_)
