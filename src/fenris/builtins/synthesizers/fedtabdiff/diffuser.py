"""
FedTabDiff Diffuser.

From https://github.com/sattarov/FedTabDiff/blob/main/BaseDiffuser.py.
"""

from typing import Any

import torch
from torch import Tensor


class Diffuser:
    def __init__(
        self,
        device: str,
        total_steps: int,
        beta_start: float,
        beta_end: float,
        scheduler: str,
    ):

        self._total_steps = total_steps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self.device = device

        self.alphas, self.betas = self._prepare_noise_scheduler(scheduler)
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def _prepare_noise_scheduler(self, scheduler: str) -> tuple[Tensor, Tensor]:
        scale = 1000 / self._total_steps
        beta_start = scale * self._beta_start
        beta_end = scale * self._beta_end

        match scheduler:
            case "linear":
                betas = torch.linspace(
                    beta_start,
                    beta_end,
                    self._total_steps,
                )
            case "quad":
                betas = (
                    torch.linspace(
                        self._beta_start**0.5,
                        self._beta_end**0.5,
                        self._total_steps,
                    )
                    ** 2
                )
            case _:
                raise ValueError(f"Unknown scheduler {scheduler}")

        alphas = 1.0 - betas
        return alphas.to(self.device), betas.to(self.device)

    def sample_timesteps(self, n: int, generator: torch.Generator) -> Tensor:
        """Sample random timesteps.

        Args:
            n (int): number of timesteps
            generator (torch.Generator): local generator for reproducibility

        Returns:
            Tensor: Sampled timesteps
        """
        t = torch.randint(
            low=1,
            high=self._total_steps,
            size=(n,),
            device=self.device,
            generator=generator,
        )
        return t

    def add_gauss_noise(
        self, x_num: Tensor, timesteps: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        """Add gaussian noise to the input data given a specific timestep
        value.

        Args:
            x_num (Tensor): input data tensor
            timesteps (Tensor): list of timesteps
            generator (torch.Generator): local generator for reproducibility

        Returns:
            tuple[Tensor, Tensor]
        """
        # numeric attributes
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[timesteps])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[timesteps])[:, None]
        noise_num = torch.randn(
            x_num.shape, dtype=x_num.dtype, device=x_num.device, generator=generator
        )
        x_noise_num = sqrt_alpha_hat * x_num + sqrt_one_minus_alpha_hat * noise_num
        return x_noise_num, noise_num

    def p_sample_gauss(
        self,
        model_out: Any,
        z_norm: Tensor,
        timesteps: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        """
        Sampling or denoising step.

        Args:
            model_out: trained model used for noise removal
            z_norm (Tensor): initial data tensor
            timesteps (Tensor): timesteps
            generator (torch.Generator): local generator for reproducibility

        Returns:
            Tensor: denoised tensor
        """
        sqrt_alpha_t = torch.sqrt(self.alphas[timesteps])[:, None]
        betas_t = self.betas[timesteps][:, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alphas_hat[timesteps])[:, None]
        epsilon_t = torch.sqrt(self.betas[timesteps][:, None])

        random_noise = torch.randn(
            z_norm.shape, dtype=z_norm.dtype, device=z_norm.device, generator=generator
        )
        random_noise[timesteps == 0] = 0.0

        model_mean = (1 / sqrt_alpha_t) * (
            z_norm - (betas_t * model_out / sqrt_one_minus_alpha_hat_t)
        )
        z_norm = model_mean + (epsilon_t * random_noise)

        return z_norm
