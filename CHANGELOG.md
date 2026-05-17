# Changelog

This file documents all FENRIS releases.
The format is based on [Common Changelog](https://common-changelog.org/).<br>
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-rc.1] - 2026-05-18

### Added

- Implement core benchmarking pipeline with configurable federated
  training, evaluation, and artifact generation
- Introduce registry-based plugin architecture for synthesizers,
  coordinators, partitioners, and evaluators
- Add built-in synthesizers: Fed-TGAN, FedTabDiff, FedSimpleGAN
- Add built-in coordinators: FedAvg, Fed-TGAN coordinator
- Add built-in partitioners: IID, Linear, Square, Exponential, Dirichlet,
  Pathological, Shard, and Continuous (via flwr-datasets)
- Add built-in evaluators covering fidelity, utility, privacy, fairness,
  and scalability metric categories
- Add FedHello (minimal reference synthesizer) and FedNaughty
  (chaos-testing synthesizer for fault injection)
- Add patience-based early stopping with configurable metric, direction,
  and patience parameters
- Support schema inference and user-provided schema files
- Add scaffolding system for generating new plugin projects and component stubs
- Derive deterministic seeds from a single master seed for reproducibility
- Record platform metadata and component metadata per run
- Add smoke test suite runnable via pytest
- Add CI pipeline with Ruff, mypy, and pytest enforcement
- Add user and developer documentation