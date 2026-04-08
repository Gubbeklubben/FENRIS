"""
Generator with Residual blocks for Fed-TGAN.

Ported from the original Fed-TGAN implementation:
https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/synthesizers/ctgan.py

Residual blocks use skip connections to improve gradient flow and model capacity.
"""

import torch
from torch import Tensor, nn


class Residual(nn.Module):  # type: ignore[misc]
    """Residual block with skip connection.

    Input: (batch, input_dim)
    Output: (batch, input_dim + output_dim)  # Concatenates input with output
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(Residual, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        # Concatenate output with input (skip connection)
        return torch.cat([out, x], dim=1)


class Generator(nn.Module):  # type: ignore[misc]
    def __init__(
        self, latent_dim: int, output_dim: int, gen_dims: tuple[int, int] = (256, 256)
    ):
        """Generator with Residual blocks.

        Parameters
        ----------
        latent_dim : int
            Input latent dimension (noise + conditional)
        output_dim : int
            Output dimension (encoded data dimension)
        gen_dims : tuple[int, int]
            Hidden dimensions for residual blocks (default: (256, 256))
        """
        super(Generator, self).__init__()

        # Build sequence of residual blocks
        layers = []
        dim = latent_dim
        for hidden_dim in gen_dims:
            layers.append(Residual(dim, hidden_dim))
            dim += hidden_dim  # Dimension accumulates due to skip connections

        # Final linear layer to output dimension
        layers.append(nn.Linear(dim, output_dim))

        self.seq = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        # No activation at the end
        # apply_activate() handles column-specific activations
        return self.seq(z)
