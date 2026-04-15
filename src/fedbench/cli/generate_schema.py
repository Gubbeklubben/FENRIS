import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from fedbench.core.data import load_csv
from fedbench.core.data.schemas import infer_schema

app = typer.Typer()


@app.command(help="Infer a fixed schema from the input dataset and write it to file.")
def generate_schema(
    dataset_file: Annotated[
        Path,
        typer.Argument(
            help="Path to dataset in CSV format.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    schema_file: Annotated[
        Path | None,
        typer.Option(
            help="Override the schema file output path. Defaults to a .schema.json "
            "file next to the input dataset. File must not already exist.",
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite the schema file if it already exists."),
    ] = False,
) -> None:
    schema_path = schema_file or dataset_file.with_suffix(".schema.json")

    if not force and schema_path.exists():
        raise typer.BadParameter(
            f"Cannot write schema file: `{schema_path}` already exists. "
            "Use --force to overwrite the existing schema file."
        )

    df = load_csv(dataset_file)
    schema = infer_schema(df)

    with schema_path.open("w") as f:
        json.dump(asdict(schema), f, indent=4)

    print(f"Inferred schema for `{dataset_file}` written to `{schema_path}`.")
