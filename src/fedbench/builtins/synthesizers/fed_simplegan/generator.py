import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Generator(nn.Module):  # type: ignore[misc]
    def __init__(self, latent_dim: int, output_dim: int):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(
            64, output_dim
        )  # Output dimension matches tabular data features

    def forward(self, z: Tensor) -> Tensor:
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        # Sigmoid activation to constrain output to [0, 1] range
        # This matches our MinMaxScaler normalization
        return torch.sigmoid(z)
