import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .diffuser import Diffuser
from .flwr_client import get_client_fn, get_eval_config
from .mlpsynth import MLPSynthesizer
from .utils import get_parameters


def init_model(cfg):
    """ Initialize model

    Args:
        cfg (dict): experiment parameters

    Returns:
        synthesizer: synthesizer model
        diffuser: diffuser model
    """
    log(INFO, f"Initializing FedTabDiff model")
    # define synthesizer
    synthesizer = MLPSynthesizer(
        d_in=cfg['encoded_dim'],
        hidden_layers=cfg['mlp_layers'],
        activation=cfg['activation'],
        n_cat_tokens=cfg['n_cat_tokens'],
        n_cat_emb=cfg['n_cat_emb'],
        n_classes=cfg['n_classes'],
        embedding_learned=False
    )
    # define diffuser
    diffuser = Diffuser(
        total_steps=cfg['diffusion_steps'],
        beta_start=cfg['diffusion_beta_start'],
        beta_end=cfg['diffusion_beta_end'],
        device=cfg['device'],
        scheduler=cfg['scheduler'])

    return synthesizer, diffuser


def train_model(synthesizer, diffuser, train_loader, cfg,
                optimizer=None):
    """ Training function for FedTabDiff

    Args:
        synthesizer: synthesizer model
        diffuser: diffuser model
        train_loader (torch data loader): training data loader
        cfg (dict): experiment parameters
        optimizer (torch optimizer, optional): optimizer. Defaults to None.

    Returns:
        float: training loss
    """
    device = cfg['device']
    client_rounds = cfg['client_rounds']

    # init optimizer
    parameters = filter(lambda p: p.requires_grad, synthesizer.parameters())
    if optimizer is None:
        optimizer = optim.Adam(parameters, lr=cfg['learning_rate'])

    # init loss function
    loss_fnc = nn.MSELoss()
    total_losses = []

    # iterate over client rounds
    round = 0

    # iterate over distinct mini-batches
    for _, (batch_cat, batch_num, batch_y) in enumerate(train_loader):

        # set network in training mode
        synthesizer.train()
        synthesizer.to(device)

        # move batch to device
        batch_cat = batch_cat.to(device)
        batch_num = batch_num.to(device)
        batch_y = batch_y.to(device)

        # sample timestamps t
        timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])

        # get cat embeddings
        batch_cat_emb = synthesizer.embed_categorical(x_cat=batch_cat)

        # concat cat & num
        batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

        # add noise
        batch_noise_t, noise_t = diffuser.add_gauss_noise(x_num=batch_cat_num,
                                                          t=timesteps)

        # conduct forward encoder/decoder pass
        predicted_noise = synthesizer(x=batch_noise_t, timesteps=timesteps,
                                      label=batch_y)

        # compute train loss
        train_losses = loss_fnc(
            input=noise_t,
            target=predicted_noise,
        )

        # reset encoder and decoder gradients
        optimizer.zero_grad()

        # run error back-propagation
        train_losses.backward()

        # optimize encoder and decoder parameters
        optimizer.step()

        # collect rec error losses
        total_losses.append(train_losses.detach().cpu().numpy())

        round += 1
        if round >= client_rounds:
            break

    # average of rec errors
    total_losses_mean = np.mean(np.array(total_losses)).item()

    return total_losses_mean


@torch.no_grad()
def generate_samples(
        synthesizer,
        diffuser,
        encoded_dim,
        last_diff_step,
        n_samples=None,
        label=None
):
    """ Generation of samples.
        For unconditional sampling use n_samples, for conditional sampling provide label.

    Args:
        synthesizer (_type_): synthesizer model
        diffuser (_type_): diffuzer model
        encoded_dim (int): transformed data dimension
        last_diff_step (int): total number of diffusion steps
        n_samples (int, optional): number of samples to sample. Defaults to None.
        label (tensor, optional): list of labels for conditional sampling. Defaults to None.

    Returns:
        torch.Tensor: matrix of generated samples
    """
    device = next(synthesizer.parameters()).device
    if (n_samples is None) and (label is None):
        raise Exception("either n_samples or label needs to be given")

    if label is not None:
        n_samples = len(label)

    # initialize noise
    z_norm = torch.randn((n_samples, encoded_dim)).float()

    label = label.to(device)
    z_norm = z_norm.to(device)

    # iterate over diffusion steps
    pbar = tqdm(iterable=reversed(range(0, last_diff_step)))
    for i in pbar:
        # update progress bar
        pbar.set_description(f"SAMPLING STEP: {i:4d}")

        # sample timestamps t
        t = torch.full((n_samples,), i, dtype=torch.long).to(device)

        # conduct forward encoder/decoder pass
        model_out = synthesizer(z_norm, t, label)

        # reverse diffusion step, i.e. noise removal
        z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

    return z_norm


def decode_samples(
        samples,
        cat_dim,
        n_cat_emb,
        num_attrs,
        cat_attrs,
        num_scaler,
        vocab_per_attr,
        label_encoder,
        embeddings,
):
    """ Decoding function for unscaling numeric attributes and inverse encoding of categorical attributes.
        Used once synthetic data is generated.

    Args:
        sample (tensor): input samples for decoding
        cat_dim (int): categorical dimension
        n_cat_emb (int): size of categorical embeddings
        num_attrs (list): numeric attributes
        cat_attrs (list): categorical attributes
        num_scaler (_type_): numeric scaler from sklearn
        vocab_per_attr (dict): vocabulary of distinct values in attribute
        label_encoder (_type_): categorical encoder
        embeddings (_type_): embeddings

    Returns:
        pandas DataFrame: decoded samples
    """

    # split sample into numeric and categorical parts
    # samples = samples.cpu().numpy()
    samples_num = samples[:, cat_dim:]
    samples_cat = samples[:, :cat_dim]

    # denormalize numeric attributes
    z_norm_upscaled = num_scaler.inverse_transform(samples_num.cpu().numpy())
    z_norm_df = pd.DataFrame(z_norm_upscaled, columns=num_attrs)

    # reshape back to batch_size * n_dim_cat * cat_emb_dim
    samples_cat = samples_cat.reshape(-1, len(cat_attrs), n_cat_emb)

    # compute batch-wise calculation of distances because for datasets with large number of embedding tokens can be memory costly
    batch_size = 2048
    n_samples = len(samples)
    z_cat_df_list = []

    # iterate over generated categorical samples
    for i in range(0, n_samples, batch_size):
        # get batch of samples
        samples_cat_subset = samples_cat[i: i + batch_size]

        # compute pairwise distances between embeddings and generated samples
        distances = torch.cdist(x1=embeddings, x2=samples_cat_subset)

        # create temp dataframes for collection of intermediate results
        z_cat_df_temp = pd.DataFrame(index=range(len(samples_cat_subset)),
                                     columns=cat_attrs)

        for attr_idx, attr_name in enumerate(cat_attrs):
            # get vocab indices for attribute
            attr_emb_idx = list(vocab_per_attr[attr_name])

            # get distances for attribute
            attr_distances = distances[:, attr_emb_idx, attr_idx]

            # get nearest embedding index
            _, nearest_idx = torch.min(attr_distances, dim=1)

            # convert to numpy
            nearest_idx = nearest_idx.cpu().numpy()

            # map emb indices back to column indices
            z_cat_df_temp[attr_name] = np.array(attr_emb_idx)[nearest_idx]

        # collect temp DFs
        z_cat_df_list.append(z_cat_df_temp)

    # concat DFs
    z_cat_df = pd.concat(z_cat_df_list, ignore_index=True)

    # inverse transform categorical attributes
    z_cat_df = z_cat_df.apply(label_encoder.inverse_transform)

    # concat numeric and categorical attributes
    sample_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)

    return sample_decoded


def main(cfg):
    rng = np.random.default_rng(cfg['seed'])  # set numpy seed
    torch.manual_seed(cfg['seed'])  # set pytorch seed CPU
    torch.cuda.manual_seed(cfg['seed'])  # set pytorch seed GPU

    #################### Load and preprocess dataset ####################
    # The City of Philadelphia Payments is used for showing the technique.
    # The dataset can be accessed under the following link: https://www.phila.gov/2019-03-29-philadelphias-initial-release-of-city-payments-data

    # read csv file
    train_raw = pd.read_csv(r'data/city_payments_fy2017.csv.zip')

    # remove underscore in column names for correct inverse decoding
    train_raw.columns = [col.replace('_', ' ') for col in train_raw.columns]

    # identify numeric and categorical attributes
    cat_attrs = ['fm', 'check date', 'department title', 'character title',
                 'sub obj title', 'vendor name', 'contract description']
    num_attrs = ['transaction amount']

    # extract label
    label_name = 'doc ref no prefix definition'
    label = train_raw[label_name]

    # take subset of top 5 most frequent label values
    top_n = train_raw[label_name].value_counts().nlargest(5).index
    train_raw = train_raw[train_raw[label_name].isin(top_n)].reset_index(
        drop=True)

    # add col name to every entry to make them distinguishable for embedding
    for cat_attr in cat_attrs:
        train_raw[cat_attr] = cat_attr + '_' + train_raw[cat_attr].astype('str')

    # extract and transform label
    label = train_raw[label_name].fillna('NA')
    class_encoder = LabelEncoder().fit(label)
    label = class_encoder.transform(label)

    # take cat and num subsets
    train = train_raw[[*cat_attrs, *num_attrs]]

    # update categorical attributes
    train[cat_attrs] = train[cat_attrs].astype(str)

    print('Processed data: Train shape: {}'.format(train.shape))

    ### transform numeric attributes
    num_scaler = QuantileTransformer(output_distribution='normal',
                                     random_state=cfg['seed'])
    num_scaler.fit(train[num_attrs])
    train_num_scaled = num_scaler.transform(train[num_attrs])

    ### transform categorical attributes
    # get unique vocabulary values
    vocabulary_classes = np.unique(train[cat_attrs])
    # fit label encoder
    label_encoder = LabelEncoder().fit(vocabulary_classes)
    # transform dataset
    train_cat_scaled = train[cat_attrs].apply(label_encoder.transform)
    # collect unique categories of each attribute
    vocab_per_attr = {cat_attr: set(train_cat_scaled[cat_attr]) for cat_attr in
                      cat_attrs}

    # add processed data parameters to experiment parameters
    cfg['n_cat_tokens'] = len(vocabulary_classes)
    cfg['n_classes'] = len(np.unique(label))
    cfg['cat_dim'] = cfg['n_cat_emb'] * len(cat_attrs)
    cfg['encoded_dim'] = cfg['cat_dim'] + len(num_attrs)
    cfg['vocab_per_attr'] = vocab_per_attr
    cfg['num_scaler'] = num_scaler
    cfg['num_attrs'] = num_attrs
    cfg['cat_attrs'] = cat_attrs
    cfg['label_encoder'] = label_encoder

    # init torch tensors
    train_num_torch = torch.FloatTensor(train_num_scaled)
    train_cat_torch = torch.LongTensor(train_cat_scaled.values)
    label_torch = torch.LongTensor(label)

    print('Encoded categorical data: Train shape: {}'.format(
        train_cat_torch.shape))
    print(
        'Encoded numerical data: Train shape: {}'.format(train_num_torch.shape))

    ################### Split preprocessed dataset into train and test data loaders.  ###################
    # In addition, every data loader contains multiple non-overlaping data partitions.
    # The data is partitioned according to a selected label.
    # Each partition will be assigned to individual client during training. Such scheme simulates the non-iid data split.

    # collect list of indices based on label (non-iid splits)
    unique_keys = np.unique(label)
    data_split_mapping = {k: np.argwhere(label == k).squeeze() for k in
                          unique_keys}

    # split train, test and label sets
    train_loaders_client = []
    test_loaders_client = []
    test_loader_server = (train,
                          label_torch)  # here the complete train set is used to evaluate on the entire population distribution

    for indices in data_split_mapping.values():
        # pack train partitiones into TensorDataset
        train_set = TensorDataset(
            train_cat_torch[indices],
            train_num_torch[indices],
            label_torch[indices]
        )

        # pack test partitiones
        test_set = (
            train.iloc[indices],
            label_torch[indices]
        )

        # append train and test loaders
        train_loaders_client.append(
            DataLoader(train_set, batch_size=cfg['batch_size'],
                       shuffle=True))
        test_loaders_client.append(test_set)

    #################### Initialize synthesizer (FinDiff) and flower client/server functions ###################

    # init synthesizer and diffuser
    synthesizer, diffuser = init_model(cfg=cfg)

    # Get the initialized model parameters
    init_params = get_parameters(synthesizer)

    # define client function. It will be called by the VirtualClientEngine whenever a client is sampled by the strategy to participate.
    client_fn = get_client_fn(
        train_loaders=train_loaders_client,
        test_loaders=test_loaders_client,
        cfg=cfg
    )

    # define server function for evaluation of entire population. It will be called at every training round.
    evaluate_server_fn = get_evaluate_server_fn(
        test_loader=test_loader_server,
        cfg=cfg
    )

    # init strategy parameters
    strategy_params = dict(
        fraction_fit=cfg['fraction_fit'],
        fraction_evaluate=cfg['fraction_evaluate'],
        min_fit_clients=cfg['min_fit_clients'],
        min_evaluate_clients=cfg['min_evaluate_clients'],
        min_available_clients=cfg['n_clients'],
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
        evaluate_fn=evaluate_server_fn,
        on_fit_config_fn=get_eval_config,
        on_evaluate_config_fn=get_eval_config
    )
    strategy = FedAvg(**strategy_params)  # FedAvg is the default strategy

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if cfg['device'] == 'cuda':
        client_resources = {"num_gpus": 1, "num_cpus": 4}

    #################### Start simulation ###################

    # Start simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg['n_clients'],
        config=fl.server.ServerConfig(num_rounds=cfg['server_rounds']),
        strategy=strategy,
        client_resources=client_resources
    )
