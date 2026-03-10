# This file is ripped from https://github.com/sattarov/FedTabDiff/tree/main
# and has not been significantly modified.
import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def init_linear_layer(input_size: int, hidden_size: int) -> nn.Linear:
    linear = nn.Linear(input_size, hidden_size, bias=True)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.constant_(linear.bias, 0.0)
    return linear


def timestep_embedding(
    timesteps: Tensor,
    dim_out: int,
    max_period: int = 10000,
) -> Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim_out: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim_out // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim_out % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLP(nn.Module):  # type: ignore[misc]
    """Base FeedForward Network"""

    def __init__(
        self,
        hidden_size: Sequence[int],
        activation: str = "lrelu",
    ) -> None:

        super(MLP, self).__init__()
        # init encoder architecture
        self.layers = self.init_layers(hidden_size)
        if activation == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def init_layers(self, layer_dimensions: Sequence[int]) -> list[nn.Linear]:

        layers = []
        for i in range(len(layer_dimensions) - 1):
            linear_layer = init_linear_layer(
                layer_dimensions[i], layer_dimensions[i + 1]
            )
            layers.append(linear_layer)

            self.add_module("linear_" + str(i), linear_layer)
        return layers

    def forward(self, x: Tensor) -> Tensor:
        # Define the forward pass
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        return x


class MLPSynthesizer(nn.Module):  # type: ignore[misc]
    """Feed Forward Network used as a synthesizer in the diffusion process."""

    def __init__(
        self,
        d_in: int,
        hidden_layers: Sequence[int],
        activation: str = "lrelu",
        dim_t: int = 64,
        n_cat_tokens: int | None = None,
        n_cat_emb: int | None = None,
        embedding: Tensor | None = None,
        embedding_learned: bool = True,
        n_classes: int | None = None,
    ) -> None:
        """
        Constructor for initializing the synthesizer

        Args:
            d_in (int): dimensionality of the input data
            hidden_layers (list): list of the neurons in every hidden layer
            activation (str, optional): activation function. Defaults to 'lrelu'.
            dim_t (int, optional): Dimensionality of the intermediate layer for
                connecting embeddings. Defaults to 64.
            n_cat_tokens (int, optional): Number of total categorical tokens.
                Defaults to None.
            n_cat_emb (int, optional): Dim of categorical embeddings.
                Defaults to None.
            embedding (tensor, optional): Provide if learned embeddings are given.
                Defaults to None.
            embedding_learned (bool, optional): Flag whether embeddings need to be
                learned. Defaults to True.
            n_classes (int, optional): Total number of classes, if conditional
                sampling is required. Defaults to None.
        """
        super(MLPSynthesizer, self).__init__()
        self.dim_t = dim_t
        self.mlp = MLP([dim_t, *hidden_layers], activation=activation)
        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=embedding)
        elif n_cat_tokens and n_cat_emb:
            self.embedding = nn.Embedding(
                n_cat_tokens,
                n_cat_emb,
                max_norm=None,
                scale_grad_by_freq=False,
            )
            self.embedding.weight.requires_grad = embedding_learned

        # embed label
        if n_classes is not None:
            self.label_emb = nn.Embedding(n_classes, dim_t)

        # projection used for the input data
        self.proj = nn.Sequential(
            nn.Linear(d_in, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        # projection for the time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        # used for the output layer
        self.head = nn.Linear(hidden_layers[-1], d_in)

    def get_embeddings(self) -> Tensor:
        """Extract embedding vectors

        Returns:
            tensor: embedding vectors
        """
        return self.embedding.weight.data

    def embed_categorical(self, x_cat: Tensor) -> Tensor:
        """Perform embedding mapping for categorical attributes

        Args:
            x_cat (tensor): categorical tokens

        Returns:
            tensor: embeddings
        """

        # perform embedding mapping and then reshape
        x_cat_emb = self.embedding(x_cat)
        x_cat_emb = x_cat_emb.view(-1, x_cat_emb.shape[1] * x_cat_emb.shape[2])
        return x_cat_emb

    def forward(
        self, x: Tensor, timesteps: Tensor, label: Tensor | None = None
    ) -> Tensor:

        # time embeddings
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))

        # add label embeddings
        if label is not None:
            emb = emb + self.label_emb(label)

        # aggregated data projection with time & label embeddings
        x = self.proj(x) + emb

        # additional mlp layers
        x = self.mlp(x)
        x = self.head(x)
        return x
