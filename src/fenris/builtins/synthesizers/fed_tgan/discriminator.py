"""PacGAN Discriminator for Fed-TGAN.

Ported from the original Fed-TGAN implementation:
https://github.com/zhao-zilong/Fed-TGAN/blob/main/Server/dtds/synthesizers/ctgan.py

PacGAN (Packing GAN) packs multiple samples together to help prevent mode collapse.
"""

from torch import Tensor, nn


class Discriminator(nn.Module):  # type: ignore[misc]
    """PacGAN Discriminator with sample packing.

    Parameters
    ----------
    input_dim : int
        Dimension of each input sample
    dis_dims : tuple[int, int]
        Hidden layer dimensions (default: (256, 256))
    pack : int
        Number of samples to pack together (default: 10)
    """

    def __init__(
        self, input_dim: int, dis_dims: tuple[int, int] = (256, 256), pack: int = 10
    ):
        super().__init__()
        self.pack = pack
        self.packdim = input_dim * pack

        # Build sequence with LeakyReLU and Dropout
        layers = []
        dim = self.packdim
        for hidden_dim in dis_dims:
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
            ])
            dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(dim, 1))

        self.seq = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sample packing.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, input_dim)
            batch_size must be divisible by pack

        Returns
        -------
        Tensor
            Discriminator output (logits, not probabilities)
        """
        assert x.size(0) % self.pack == 0, (
            f"Batch size must be divisible by pack={self.pack}"
        )
        # Reshape to pack samples: (batch/pack, pack*input_dim)
        return self.seq(x.view(-1, self.packdim))
