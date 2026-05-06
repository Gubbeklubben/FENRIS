from typing import Annotated, Literal

import typer

import fenris.app.run.runner as runner
from fenris.app.run.pipeline import pipeline
from fenris.config.builder import build_config
from fenris.config.parsing import parse_args, parse_kwargs

app = typer.Typer()


@app.command()
def run(
    synthesizer: Annotated[str, typer.Argument(help="Synthesizer name.")],
    coordinator: Annotated[str, typer.Argument(help="Coordinator name.")],
    partitioner: Annotated[str, typer.Argument(help="Partitioner name.")],
    dataset: Annotated[str, typer.Argument(help="Path to the dataset CSV.")],
    synthesizer_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the synthesizer (key=value)."
        ),
    ] = None,
    coordinator_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the coordinator (key=value)."
        ),
    ] = None,
    partitioner_kwargs: Annotated[
        str | None,
        typer.Option(
            callback=parse_kwargs, help="Kwargs for the partitioner (key=value)."
        ),
    ] = None,
    target_col: Annotated[
        str | None, typer.Option(help="Target column for utility/fairness metrics.")
    ] = None,
    sensitive_cols: Annotated[
        str | None,
        typer.Option(
            callback=parse_args,
            help="Comma-separated list of sensitive columns for fairness/AIA metrics.",
        ),
    ] = None,
    run_categories: Annotated[
        str | None,
        typer.Option(
            callback=parse_args, help="Override specific metric categories to run."
        ),
    ] = None,
    early_stop: Annotated[
        bool | None, typer.Option(help="Enable threshold-based convergence.")
    ] = None,
    stop_metric: Annotated[
        str | None,
        typer.Option(help="Metric key to monitor (e.g., `fidelity.corr_fro_diff`)."),
    ] = None,
    stop_mode: Annotated[
        Literal["min", "max"] | None,
        typer.Option(
            help="Whether stop metric is expected to converge "
            "towards a minimum or a maximum."
        ),
    ] = None,
    stop_epsilon: Annotated[
        float | None, typer.Option(help="Minimum improvement required.")
    ] = None,
    stop_patience: Annotated[
        int | None,
        typer.Option(help="Evaluations without improvement before stopping."),
    ] = None,
    stop_min_rounds: Annotated[
        int | None, typer.Option(help="Minimum rounds before stopping.")
    ] = None,
    stop_eval_every: Annotated[
        int | None, typer.Option(help="Evaluate stopping metric every N rounds.")
    ] = None,
    stop_synthetic_rows: Annotated[
        int | None, typer.Option(help="Rows sampled for early stopping evaluation.")
    ] = None,
    num_clients: Annotated[
        int | None, typer.Option(help="Number of simulated clients.")
    ] = None,
    num_rounds: Annotated[
        int | None, typer.Option(help="Maximum number of federated rounds.")
    ] = None,
    test_size: Annotated[
        float | None,
        typer.Option(
            help="Fraction of data to hold out for testing. "
            "This is used for both local and global holdout fractions."
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Master random seed.")] = None,
    outputdir: Annotated[
        str | None, typer.Option(help="Output directory for artifacts.")
    ] = None,
    num_synthetic_rows: Annotated[
        int | None, typer.Option(help="Number of synthetic rows to generate.")
    ] = None,
    client_cpus: Annotated[
        float | None,
        typer.Option(
            help="Number of CPUs to use per client. If the total CPU requirement "
            "exceeds what is available, some operations will execute sequentially "
            "rather than in parallel."
        ),
    ] = None,
    client_gpus: Annotated[
        float | None,
        typer.Option(
            help="Number of GPUs to use per client. If the total GPU requirement "
            "exceeds what is available, some operations will execute sequentially "
            "rather than in parallel."
        ),
    ] = None,
    disable_pickle: Annotated[
        bool | None, typer.Option(help="Disable pickle based serialization.")
    ] = None,
    schema: Annotated[
        str | None,
        typer.Option(
            help="Path to the fixed schema to use for evaluation (in FenrisSchema "
            "format). If not specified, will look for a .schema.json file with the "
            "same base name as the dataset file. If this does not exist, a schema "
            "will be inferred."
        ),
    ] = None,
) -> None:

    cli_input = {
        key: value  # nofmt
        for key, value in locals().items()
        if value is not None
    }
    config = build_config(cli_input)
    runner.run(config, pipeline())
