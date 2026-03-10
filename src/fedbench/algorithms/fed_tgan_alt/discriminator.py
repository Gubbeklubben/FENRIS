"""Fed-TGAN Discriminator network.

PacGAN-style discriminator following CTGAN's architecture.
Uses WGAN-GP loss (no sigmoid in the output).
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):  # type: ignore[misc]
    """Discriminator for Fed-TGAN / CTGAN with PacGAN.

    Parameters
    ----------
    input_dim
        Dimension of encoded data + conditional vector.
    discriminator_dim
        Sequence of hidden layer sizes.
    pac
        Number of samples packed together (PacGAN).
    """

    def __init__(
        self,
        input_dim: int,
        discriminator_dim: Sequence[int],
        pac: int = 10,
    ) -> None:
        super().__init__()
        self.pac = pac
        self.pacdim = input_dim * pac

        dim = self.pacdim
        seq: list[nn.Module] = []
        for item in discriminator_dim:
            seq.extend([nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)])
            dim = item
        seq.append(nn.Linear(dim, 1))
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(
        self,
        real_data: Tensor,
        fake_data: Tensor,
        device: torch.device | str = "cpu",
        lambda_: float = 10.0,
    ) -> Tensor:
        """Compute the WGAN-GP gradient penalty.

        Parameters
        ----------
        real_data
            Real data batch (packed by ``pac``).
        fake_data
            Generator output batch (packed by ``pac``).
        device
            Torch device for intermediate tensors.
        lambda_
            Penalty coefficient (default 10).

        Returns
        -------
        Tensor
            Scalar gradient penalty loss.
        """
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1, device=device)
        alpha = alpha.repeat(1, self.pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad_(True)

        # Disable dropout so the gradient penalty is deterministic
        was_training = self.training
        self.eval()
        disc_interpolates = self(interpolates)
        if was_training:
            self.train()

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = (
            gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
        )
        gradient_penalty = (gradients_view**2).mean() * lambda_
        return gradient_penalty

    def forward(self, input_: Tensor) -> Tensor:
        """Discriminate packed data.

        Parameters
        ----------
        input_
            Packed or unpacked data tensor.

        Returns
        -------
        Tensor
            Discriminator score (1-D).
        """
        return self.seq(input_.view(-1, self.pacdim))
