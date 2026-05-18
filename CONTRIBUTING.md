# Contributing to FENRIS

This guide describes how to set up a development environment and contribute to the FENRIS framework.

## Contribution policy

Contributions are welcome. For non-trivial changes, open an issue to discuss the proposed change before submitting a pull request. Pull requests should target the `main` branch and include a clear description of the change. At least one approving review is required before merging. All communication with maintainers &ndash; issue descriptions, pull request bodies, and review responses &ndash; must be written in the contributor's own words. Contributors are expected to understand and be able to explain any code submitted.

## Acquiring the source code

Clone the repository and navigate to the project root:

```bash
git clone https://github.com/Gubbeklubben/FENRIS.git
cd FENRIS
```

## Environment setup

FENRIS uses [Poetry](https://python-poetry.org/) for dependency management. If Poetry is not already installed, the recommended method is via `pipx`:

```bash
sudo apt install pipx
pipx install poetry
# (restart shell to apply PATH change)
poetry self update
```

To create a virtual environment and install all dependencies:

```bash
poetry install
```

FENRIS supports Python 3.12 and 3.13. To change the virtual environment's Python version:

```bash
poetry env use 3.12
```

Activate the virtual environment:

```bash
# Linux/macOS (including WSL)
eval $(poetry env activate)

# Windows
Invoke-Expression (poetry env activate)
```

Alternatively, prefix individual commands with `poetry run` without activating the environment:

```bash
poetry run fenris --version
```

To verify the setup:

```bash
fenris --version
```

## Development workflow

Before opening a pull request, ensure all of the following checks pass locally &ndash; they are also enforced in CI.

**Formatting and linting** (Ruff):

```bash
ruff format src tests
ruff check src tests --fix
```

**Type checking** (mypy):

```bash
mypy -p fenris
```

**Tests** (pytest):

```bash
pytest tests
```

Pre-commit hooks can be installed to run Ruff automatically before every commit:

```bash
pre-commit install
```
