# FedBench

An Extensible Benchmarking Framework for Federated Synthetic Tabular Data Generators

## Quickstart (WIP)

### Setting up the virtual environment

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

All commands below will assume you are running them from the project root
and that you have activated the Poetry virtual environment.

To activate the venv on *nix:

```bash
eval $(poetry env activate)
```

On Windows (not recommended with Flower's Ray backend):

```powershell
Invoke-Expression (poetry env activate)
```

If you have not activated the Poetry virtual environment for whatever reason,
all Fedbench commands must be prefixed with `poetry run` to execute correctly.

### Listing available components
To list all available components:

```bash
fedbench show
```

To list only certain types of components, specify the type(s) as an argument:

```bash
fedbench show evaluators
fedbench show partitioners
fedbench show synthesizers coordinators
```

### Running a benchmarking pipeline

Minimal example pipeline run:

```bash
fedbench run \
  fed_hello fedavg iid_partitioner datasets/breast_cancer.csv
```

Example pipeline run with FedTabDiff:

```bash
fedbench run \
  fedtabdiff fedavg iid_partitioner datasets/breast_cancer.csv \
  --synthesizer-kwargs "\
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

The synthesizer kwargs do not need to be explicitly specified. They are shown here with their default values for illustration purposes. Note that all synthesizer kwargs must be specified in a single comma-separated list contained within a single command line argument (meaning it needs to be quoted if it contains spaces).