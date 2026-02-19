# FedBench
An Extensible Benchmarking Framework for Federated Synthetic Tabular Data Generators

## Quickstart (WIP)
This framework uses Poetry for dependency management. If not already installed, install the latest version using your preferred method, for example:
```
sudo apt install pipx
pipx install poetry
# (restart shell to apply PATH change)
poetry self update
```

To install dependencies, navigate to the project root and run:
```
poetry install
```

The framework only supports Python 3.12 and 3.13. If you need to change the Python version for the venv:
```
poetry env use 3.12
```

Example pipeline run:
```
poetry run python -m fedbench run fed_noop iid-partitioner datasets/breast_cancer.csv --partitioner-kwargs num_partitions=3 --allow-pickle
```
