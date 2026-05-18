# Example Pipelines

All examples assume FENRIS is [installed](README.md#installation) and that the [reference datasets](README.md#acquiring-the-reference-datasets) have been downloaded to a `datasets/` subfolder. A complete CLI reference is available in [CLI.md](CLI.md).

## Smoke test

The following command uses the minimal `fed_hello` synthesizer to verify that the installation is working without running any real model:

```bash
fenris run fed_hello fedavg iid_partitioner datasets/breast_cancer.csv
```

## Basic pipeline

The following command runs a complete end-to-end benchmark pipeline on the Breast Cancer dataset using FedTabDiff. The run uses the default seed of 42 and is therefore fully reproducible:

```bash
fenris run fedtabdiff fedavg iid_partitioner datasets/breast_cancer.csv
```

## Utility and fairness metrics

Several metric families require a target column and/or sensitive columns. The following command runs FedTabDiff on the heart disease dataset with utility and fairness metrics enabled:

```bash
fenris run \
    fedtabdiff fedavg iid_partitioner datasets/heart_disease.csv \
    --target-col target \
    --sensitive-cols sex \
    --seed 42
```

## Non-IID partitioning with Fed-TGAN

Fed-TGAN requires its own coordinator. The following example also uses Dirichlet partitioning to simulate a heterogeneous, non-IID data distribution across clients:

```bash
fenris run \
    fed_tgan fed_tgan dirichlet_partitioner datasets/heart_disease.csv \
    --num-clients 5 \
    --num-rounds 20 \
    --target-col target \
    --seed 42
```

## Custom hyperparameters

The following example passes custom hyperparameters to the synthesizer. Multiple key-value pairs are separated by commas and forwarded to the synthesizer constructor:

```bash
fenris run \
    fed_simplegan fedavg iid_partitioner datasets/breast_cancer.csv \
    --synthesizer-kwargs local_epochs=10,learning_rate=0.001 \
    --num-rounds 10 \
    --seed 42
```

## Early stopping

The following example enables patience-based early stopping. Training halts once `fidelity.corr_fro_diff` does not improve for three consecutive evaluations, but not before round five:

```bash
fenris run \
    fedtabdiff fedavg iid_partitioner datasets/heart_disease.csv \
    --target-col target \
    --seed 42 \
    --early-stop \
    --stop-metric fidelity.corr_fro_diff \
    --stop-mode min \
    --stop-patience 3 \
    --stop-min-rounds 5
```
