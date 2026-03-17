from torch import Tensor, nn
import torch.nn.functional as F


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
        return self.fc3(z)  # Output synthetic tabular data
