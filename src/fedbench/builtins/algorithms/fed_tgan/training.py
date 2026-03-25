"""Utility functions for GAN training steps."""

import torch
import torch.nn as nn


def generator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    noise: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One generator update step. Returns loss."""
    generator.train()  # allow for gradient calculation
    discriminator.eval()  # prevent from updating weights

    noise = noise.to(device)
    criterion = nn.BCELoss().to(
        device
    )  # calc performance using Binary Cross Entropy Loss

    optimizer.zero_grad()  # reset gradients of optimizer

    fake_data = generator(noise)
    out_fake = discriminator(fake_data).view(
        -1
    )  # discriminator gives a tensor of probabilities

    # generator wants D(fake) -> 1
    y = torch.ones(
        out_fake.size(0), device=device
    )  # tensor of ones of same length as out_fake, target values for comparison
    loss = criterion(out_fake, y)  # compare y VS out_fake using specified loss function

    loss.backward()  # compute gradients using result from loss calc
    optimizer.step()  # update weights of synthesizer based on computed gradients

    return float(loss.item())


def discriminator_step(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One discriminator update step in (real, fake). Return loss."""
    discriminator.train()  # train mode, allow for gradient calculation

    real_data = real_data.to(device)
    fake_data = fake_data.to(device)

    y_real = torch.ones(
        real_data.size(0), device=device
    )  # target value tensor for comparison
    y_fake = torch.zeros(
        fake_data.size(0), device=device
    )  # target value tensor for comparison

    criterion = nn.BCELoss().to(device)  # define loss calculation method

    optimizer.zero_grad()

    out_real = discriminator(real_data).view(
        -1
    )  # get tensor of discriminator prediction values from real data
    out_fake = discriminator(fake_data).view(
        -1
    )  # get tensor of discriminator prediction values from fake data

    loss_real = criterion(out_real, y_real)  # calculate loss on real data probabilities
    loss_fake = criterion(out_fake, y_fake)  # calculate loss on fake data probabilities

    loss = loss_real + loss_fake  # sum of loss

    loss.backward()  # compute gradients
    optimizer.step()  # update weights

    return float(loss.item())
