# Development Commands

## Check and fix linting issues

poetry run ruff check src/ --fix \
poetry run ruff format src/

## Type checking

poetry run mypy src/

## Run tests

poetry run pytest tests/
