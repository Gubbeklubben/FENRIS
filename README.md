# FedBench
An Extensible Benchmarking Framework for Federated Synthetic Tabular Data Generators

## Quickstart (WIP)
This framework uses Poetry for dependency management. If not already installed, install the latest version using your preferred method, for example:
```
sudo apt install pipx
pipx install poetry
poetry self update
```

To install dependencies, navigate to the project root and run:
```
poetry install
```

Make sure to use Python 3.12 or 3.13 (not currently enforced by pyproject.toml - TODO). 3.14 is not supported due to flwr[simulation] dependencies.

If you need to change the Python version for the project:
```
poetry env use 3.12
```

Example pipeline run:
```
poetry run python -m fedbench run fed_noop datasets/breast_cancer.csv --num-clients 3
```
