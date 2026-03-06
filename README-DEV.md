# Development Commands

## Check and fix linting issues

This project uses Ruff for linting. By default, Ruff likes to collapse argument lists, comprehensions etc.
onto a single line, which is not always desirable. Where necessary, this can be circumvented by adding
a trailing comma to argument lists, or by placing any comment (conventionally `# nofmt`) after the first line.
If this still does not achieve desired results, it is also possible disable formatting for specific code blocks
by wrapping them in `# fmt: off` and `# fmt: on` comments. 

```bash
poetry run ruff format src
poetry run ruff check src --fix
```

## Type checking

```bash
poetry run mypy -p fedbench
```

## Run tests

```bash
poetry run pytest tests
```