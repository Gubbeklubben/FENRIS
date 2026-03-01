import torch

config = {
    "batch-size": 128,
    "max-batches": 10,
    "n-cat-emb": 2, # size of the categorical embeddings (2 means each attribute will be 2-dimensional)
    "learning-rate": 1e-4,

    "mlp-layers": [512, 512],  # total neurons at each hidden feed-forward layer
    "activation": "lrelu",

    # diffusion parameters
    "diffusion-steps": 500,
    "diffusion-beta-start": 1e-4,
    "diffusion-beta-end": 0.02,
    "scheduler": "linear",  # linear or quad

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
