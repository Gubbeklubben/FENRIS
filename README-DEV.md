# Development Commands

## Check and fix linting issues

This project uses Ruff for linting. By default, Ruff likes to collapse argument lists, comprehensions etc.
onto a single line, which is not always desirable. Where necessary, this can be circumvented by adding
a trailing comma to argument lists, or by placing any comment (conventionally `# nofmt`) after the first line.
If this still does not achieve desired results, it is also possible to disable formatting for specific code blocks
by wrapping them in `# fmt: off` and `# fmt: on` comments. 

```bash
poetry run ruff format src tests
poetry run ruff check src tests --fix
```

## Pre-commit hooks

To run ruff automatically before every commit, install the hooks once after cloning:

```bash
poetry run pre-commit install
```

To bypass the hooks in an emergency: `git commit --no-verify`

## Type checking

```bash
poetry run mypy -p fedbench
```

## Run tests

```bash
poetry run pytest tests
```

---

## Chaos Testing Algorithm (`fed_chaos`)

`fed_chaos` is a built-in algorithm that intentionally injects errors, corrupts data, and violates
protocol assumptions at configurable lifecycle points. Use it to stress-test the framework, discover
weak spots in error handling, and verify that the pipeline fails gracefully under adversarial conditions.

### Basic usage

```bash
poetry run fedbench run fed_chaos <partitioner> <dataset> \
  --algorithm-kwargs "scenario=<SCENARIO>,point=<POINT>"
```

### Parameters

| Parameter   | Type    | Default          | Description                                                 |
| :---------- | :------ | :--------------- | :---------------------------------------------------------- |
| `scenario`  | `str`   | `"crash"`        | The type of chaos to inject (see table below).              |
| `point`     | `str`   | `"synth_train"`  | Where in the lifecycle to inject (see table below).         |
| `intensity` | `float` | `1.0`            | Seconds for delay, MB for leak/large_payload. Ignored for other scenarios. |
| `exception` | `str`   | `"RuntimeError"` | Exception class for `crash` scenario.                       |

### Injection points

| Point             | Component             | Method triggered              |
| :---------------- | :-------------------- | :---------------------------- |
| `global_init`     | `FedChaos`            | `Algorithm.global_init()`     |
| `coord_fed_init`  | `FedChaosCoordinator` | `configure_fed_init()`        |
| `coord_train`     | `FedChaosCoordinator` | `aggregate_train()` / `train()` |
| `synth_fed_init`  | `FedChaosSynthesizer` | `Synthesizer.fed_init()`      |
| `synth_train`     | `FedChaosSynthesizer` | `Synthesizer.train()`         |
| `synth_sample`    | `FedChaosSynthesizer` | `Synthesizer.sample()`        |

### Scenarios

| Scenario        | What it does                                                                                   |
| :-------------- | :--------------------------------------------------------------------------------------------- |
| `crash`         | Raises a configurable exception at the target point.                                           |
| `delay`         | Blocks with `time.sleep(intensity)` seconds.                                                   |
| `leak`          | Allocates `intensity` MB of memory that is never freed.                                        |
| `corrupt`       | Returns `Update` with NaN/inf values, or `DataFrame` with wrong schema.                        |
| `wrong_type`    | Returns an object of the wrong type (e.g., `str` instead of `Update`).                         |
| `empty`         | Returns valid but logically empty objects (`Update()`, `DataFrame()`).                         |
| `infinite_loop` | Coordinator yields forever, never finishing the training round (coord_train only).              |
| `large_payload` | Allocates `intensity` MB of memory at the injection point, then raises `MemoryError`. |

### Available exception types (for `crash` scenario)

`RuntimeError`, `ValueError`, `TypeError`, `MemoryError`, `KeyboardInterrupt`, `SystemExit`,
`StopIteration`, `OverflowError`, `ZeroDivisionError`, `NotImplementedError`, `OSError`

### Example commands

```bash
# Crash during client training with a RuntimeError
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=crash,point=synth_train"

# Crash with a specific exception type
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=crash,point=synth_train,exception=MemoryError"

# 30-second delay during sampling
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=delay,point=synth_sample,intensity=30"

# Leak 100 MB during global_init
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=leak,point=global_init,intensity=100"

# Return corrupt data from the synthesizer
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=corrupt,point=synth_sample"

# Return wrong types to test type checking
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=wrong_type,point=synth_train"

# Return empty results to test semantic validation
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=empty,point=synth_sample"

# Infinite coordinator loop to test timeout handling
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=infinite_loop,point=coord_train"

# Stress serialization with a 50 MB payload
poetry run fedbench run fed_chaos iid-partitioner datasets/heart_disease.csv \
  --algorithm-kwargs "scenario=large_payload,point=synth_train,intensity=50"
```

### Interpreting results

Use the following levels to assess how well the framework handles each scenario:

| Level | Name | Description |
| :---- | :--- | :---------- |
| **0** | Vulnerable | Framework crashes ungracefully, hangs, or corrupts local state. |
| **1** | Reactive | Framework catches the error, logs it, and exits cleanly. |
| **2** | Proactive | Framework validates inputs/outputs before use and reports a specific, actionable error. |
| **3** | Resilient | Framework recovers (e.g. retries the round, skips the offending client) and completes the run. |

The current framework generally sits at **Level 1** for client-side chaos and **Level 1** for server-side chaos (Flower wraps server exceptions as a generic `RuntimeError("Exception in ServerApp thread")`). Reaching Level 2 requires input/output validation in the core runtime; Level 3 requires retry and fault-isolation logic.