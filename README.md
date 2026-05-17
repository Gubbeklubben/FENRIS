# FENRIS: Federated Extensible Norwegian Research Interface for Synthetic data

[![CI](https://github.com/Gubbeklubben/FENRIS/actions/workflows/ci.yml/badge.svg)](https://github.com/Gubbeklubben/FENRIS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fenris)](https://pypi.org/project/fenris/)
[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Example Pipelines](https://github.com/Gubbeklubben/FENRIS/blob/main/EXAMPLES.md) |
[CLI Reference](https://github.com/Gubbeklubben/FENRIS/blob/main/CLI.md) |
[Extension Guide](https://github.com/Gubbeklubben/FENRIS/blob/main/EXTENDING.md) |
[Contributing](https://github.com/Gubbeklubben/FENRIS/blob/main/CONTRIBUTING.md) |
[Changelog](https://github.com/Gubbeklubben/FENRIS/blob/main/CHANGELOG.md)

FENRIS is an extensible benchmarking framework for federated synthetic tabular data generators. Built on [Flower](https://flower.ai/), it provides a reproducible pipeline covering dataset preparation, client partitioning, federated training, multi-dimensional evaluation, and artifact generation.

The framework is designed to be extended with new algorithms, partitioning strategies and evaluation metrics through a plugin system that allows new components to be added without modifying the core system. Algorithms are conceptualized as a combination of a synthesizer (the generative model) and a coordinator (the federated training strategy), allowing for flexible combinations and comparisons.

FENRIS ships with two literature-based reference implementations (Fed-TGAN and FedTabDiff) and evaluates algorithms across fidelity, utility, privacy, fairness, and scalability.

This framework was initially developed during Spring 2026 as part of a bachelor thesis at OsloMet in collaboration with the [Norwegian Institute of Public Health](https://fhi.no/en/) (Folkehelseinstituttet, FHI) and the [Cancer Registry of Norway](https://www.fhi.no/en/cancer/cancer-registry-norway/) (Kreftregisteret).

The following sections provide a quickstart guide to using the framework, including prerequisites, setting up the environment, verifying the installation, and running a benchmarking pipeline. For more detailed documentation and examples, please refer to the links at the top of this README.

## Quickstart Guide

### Prerequisites

FENRIS v0.1.0 supports Python 3.12 and 3.13. Linux and macOS are the recommended platforms; Windows support is on a best-effort basis due to constraints imposed by Flower's Ray backend. WSL may be used as a workaround.

### Installation

The recommended installation method is `pipx`, which installs FENRIS into an isolated environment and exposes the `fenris` command globally without requiring a virtual environment:

```bash
pipx install fenris
```

To install `pipx` itself, follow the [official installation instructions](https://pipx.pypa.io/stable/how-to/install-pipx/).

Alternatively, FENRIS can be installed into a virtual environment via `pip`:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install fenris
```

To verify a successful installation:

```bash
fenris --version
```

### Acquiring the reference datasets

The reference datasets used in the examples are available at the following URLs, pinned to a stable commit:

- https://raw.githubusercontent.com/Gubbeklubben/FENRIS/6aac5c5/datasets/breast_cancer.csv
- https://raw.githubusercontent.com/Gubbeklubben/FENRIS/6aac5c5/datasets/heart_disease.csv
- https://raw.githubusercontent.com/Gubbeklubben/FENRIS/6aac5c5/datasets/lung_cancer.csv

### Executing a pipeline

The following command runs a complete end-to-end benchmark pipeline on the Breast Cancer dataset using FedTabDiff, with the default seed of 42:

```bash
fenris run fedtabdiff fedavg iid_partitioner datasets/breast_cancer.csv
```

Once the run completes, FENRIS creates a timestamped output directory, for example:

```
out/2026-05-01-14-11-26-fedtabdiff-breast_cancer-<uuid>/
```

This directory contains the following artifacts:

| File | Description |
|------|-------------|
| `synthetic.csv` | The synthetic dataset sampled from the final global model state, with the same columns as the input. |
| `metrics.centralized.json` | Metric values computed in centralized mode, where the synthetic data is evaluated against the global holdout set on the server. |
| `metrics.federated.json` | Metric values computed in federated mode, where each client evaluates the synthetic data against its local partitions and the results are aggregated server-side. |
| `config_snapshot.json` | A complete snapshot of the resolved configuration used for the run, including all default values and derived seeds. |
| `schema.json` | The resolved column schema, classifying each column by kind (`continuous`, `categorical`, `binary`, `integer`). |
| `components.json` | Metadata for each component used in the run, including plugin name, entry point, and version. |
| `platform_info.json` | A snapshot of the hardware and software environment, including OS, CPU, Python version, and relevant library versions. |

A successful run produces non-`null` fidelity metric values in both metrics files. Utility, fairness, and attribute inference metrics require `--target-col` and/or `--sensitive-cols` to be specified and will evaluate to `null` otherwise, as in this minimal example.

### Further documentation

[Example Pipelines](https://github.com/Gubbeklubben/FENRIS/blob/main/EXAMPLES.md) |
[CLI Reference](https://github.com/Gubbeklubben/FENRIS/blob/main/CLI.md) |
[Extension Guide](https://github.com/Gubbeklubben/FENRIS/blob/main/EXTENDING.md) |
[Contributing](https://github.com/Gubbeklubben/FENRIS/blob/main/CONTRIBUTING.md) |
[Changelog](https://github.com/Gubbeklubben/FENRIS/blob/main/CHANGELOG.md)