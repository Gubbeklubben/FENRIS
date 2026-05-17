# CLI Reference

The `fenris` command is invoked as:

```
fenris <command> [arguments] [options]
```

Passing `--help` to any command or subcommand prints a full description of its arguments and options. Usage examples are given in [EXAMPLES.md](EXAMPLES.md).

## Commands

| Command | Description |
|---------|-------------|
| `show` | List available components (synthesizers, coordinators, partitioners, evaluators). |
| `run` | Execute a federated benchmarking pipeline. |
| `generate-schema` | Infer a fixed schema from a CSV dataset and write it to a `.schema.json` file. |
| `plugin new` | Scaffold a new plugin project. |
| `plugin extend` | Add new component stubs to an existing plugin project. |

## `run`

The `run` command executes a complete federated benchmarking pipeline. It requires four positional arguments:

| Argument | Description |
|----------|-------------|
| `synthesizer` | Name of the installed synthesizer component. |
| `coordinator` | Name of the installed coordinator component. |
| `partitioner` | Name of the installed partitioner component. |
| `dataset` | Path to the input dataset in CSV format. |

In addition to the required arguments, `run` accepts the following options. For options that accept key-value pairs, these should be specified in the format `key1=value1,key2=value2` (optionally quoted if whitespace needs to be included).

**Component configuration**

| Option | Description |
|--------|-------------|
| `--synthesizer-kwargs` | Key-value pairs forwarded to the synthesizer constructor. |
| `--coordinator-kwargs` | Key-value pairs forwarded to the coordinator constructor. |
| `--partitioner-kwargs` | Key-value pairs forwarded to the partitioner constructor. |

**Evaluation**

| Option | Description |
|--------|-------------|
| `--target-col` | Column used as the prediction target for utility and fairness metrics, as well as any custom metric that uses a target column. |
| `--sensitive-cols` | Comma-separated column names for fairness and attribute inference metrics, as well as any custom metric that uses sensitive columns. |
| `--run-categories` | Restrict evaluation to specific metric categories; all categories run if omitted. |
| `--schema` | Path to a `.schema.json` column schema file. If omitted, looks for a file with the same name as the input dataset file; if this does not exist, a schema is inferred from the input dataset. |
| `--num-synthetic-rows` | Number of synthetic rows to generate for evaluation. |

**Federation**

| Option | Description |
|--------|-------------|
| `--num-clients` | Number of simulated clients. |
| `--num-rounds` | Maximum number of federated training rounds. |
| `--test-size` | Fraction of dataset reserved for server holdout; fraction of each client's data reserved for testing. Value should be in the interval `(0.0, 1.0)`. |
| `--seed` | Master random seed controlling all stochastic operations. |
| `--client-cpus` | Number of CPU cores allocated per client process. |
| `--client-gpus` | Number of GPUs allocated per client process. |
| `--outputdir` | Directory where run artifacts and metric results are written. |

**Early stopping**

| Option | Description |
|--------|-------------|
| `--early-stop` | Enable patience-based early stopping. |
| `--stop-metric` | Metric key to monitor (e.g. `fidelity.corr_fro_diff`). |
| `--stop-mode` | Direction of convergence: `min` or `max`. |
| `--stop-epsilon` | Minimum improvement per evaluation required to reset the patience counter. |
| `--stop-patience` | Consecutive evaluations without improvement before training stops. |
| `--stop-min-rounds` | Minimum number of rounds before early stopping may trigger. |
| `--stop-eval-every` | Evaluate the stopping metric every N rounds. |
| `--stop-synthetic-rows` | Synthetic rows sampled per early stopping evaluation. |

## `show`

The `show` command lists all installed components, grouped by type. With no arguments, all component groups are shown. To restrict the output to specific groups, one or more group names may be passed as positional arguments:

```bash
fenris show synthesizers coordinators
```

| Argument/Option | Description |
|----------------|-------------|
| `[groups...]` | Component groups to show. Valid values: `synthesizers`, `coordinators`, `partitioners`, `evaluators`. All groups are shown if omitted. |
| `--metadata` | Include factory metadata (e.g. version and author) in the output. |
| `--keywords` | Include accepted constructor parameters and their default values. |

## `generate-schema`

The `generate-schema` command inspects a CSV dataset and writes an inferred column type schema to a `.schema.json` file. The schema can then be reviewed and adjusted before being passed to `run` via `--schema`, allowing precise control over how columns are classified.

| Argument/Option | Description |
|----------------|-------------|
| `dataset_file` | Path to the input CSV file. |
| `--schema-file` | Output path for the schema file. Defaults to a `.schema.json` file placed next to the input dataset. |
| `--force` | Overwrite the schema file if it already exists. |

## `plugin`

The `plugin` group provides two subcommands for scaffolding new plugin projects and extending existing ones. See [EXTENDING.md](EXTENDING.md) for a detailed walkthrough.

| Subcommand | Description |
|-----------|-------------|
| `new` | Create a new plugin project. Accepts a project name and an optional `--parent-dir`. |
| `extend` | Add scaffold implementations to an existing plugin. Accepts the plugin path, a component group, one or more component names, and optional flags `--package`, `--base`, and `--no-group`. |
