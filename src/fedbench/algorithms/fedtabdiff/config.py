import torch

# define global experiment parameters
config = dict(
    n_cat_emb=2,
    # size of the categorical embeddings (2 means each attribute will be 2-dimensional)
    learning_rate=1e-4,  # learning rate

    mlp_layers=[512, 512],  # total neurons at each hidden feed-forward layer
    activation='lrelu',  # activation function

    # diffusion parameters
    diffusion_steps=500,  # number of diffusion steps
    diffusion_beta_start=1e-4,  # initial value of beta
    diffusion_beta_end=0.02,  # final value of beta
    scheduler='linear',  # linear or quad

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    client_rounds=10
)
