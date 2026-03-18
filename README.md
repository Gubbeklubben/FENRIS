# FedBench

An Extensible Benchmarking Framework for Federated Synthetic Tabular Data Generators

## Quickstart (WIP)

This framework uses Poetry for dependency management. If not already installed, install the latest version using your preferred method, for example:

```bash
sudo apt install pipx
pipx install poetry
# (restart shell to apply PATH change)
poetry self update
```

To install dependencies, navigate to the project root and run:

```bash
poetry install
```

The framework only supports Python 3.12 and 3.13. If you need to change the Python version for the venv:

```bash
poetry env use 3.12
```

Minimal example pipeline run:

```bash
poetry run fedbench run \
  fed_hello iid_partitioner datasets/breast_cancer.csv
```

Example pipeline run with FedTabDiff:

```bash
poetry run fedbench run \
  fed_tab_diff iid_partitioner datasets/breast_cancer.csv \
  --algorithm-kwargs "\
    batch_size=128, \
    max_batches=10, \
    n_cat_emb=2, \
    learning_rate=1e-4, \
    mlp_layers=[512,512], \
    activation=lrelu, \
    diffusion_steps=500, \
    diffusion_beta_start=1e-4, \
    diffusion_beta_end=0.02, \
    scheduler=linear \
  "
```

The algorithm kwargs do not need to be explicitly specified. They are shown here with their default values for illustration purposes. Note that all algorithm kwargs must be specified in a single comma-separated list contained within a single command line argument (meaning it needs to be quoted if it contains spaces).
