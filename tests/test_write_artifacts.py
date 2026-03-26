from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from fedbench.config import SeedConfig
from fedbench.config.config import Config, DataConfig
from fedbench.runtime.pipeline import write_artifacts


def _make_config(tmp_path: Path) -> Config:
    return Config(
        synthesizer="fed_hello",
        coordinator="MISSING",
        data=DataConfig(dataset="dummy.csv", partitioner="iid-partitioner"),
        outputdir=str(tmp_path),
        seed=SeedConfig.from_master(42),
        num_rounds=3,
        num_clients=3,
    )


def _make_ctx(
    tmp_path: Path,
    *,
    aggregated_metrics: dict[str, float] | None = None,
    centralized_metrics: dict[str, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=_make_config(tmp_path),
        run_id="test-run",
        aggregated_metrics=aggregated_metrics or {},
        centralized_metrics=centralized_metrics or {},
        synthetic_df=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    )


# --- config_snapshot.json --------------------------------------------------


def test_config_snapshot_written(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    write_artifacts(ctx)  # type: ignore[arg-type]

    snapshot = json.loads((tmp_path / "test-run" / "config_snapshot.json").read_text())
    assert snapshot["seed"] == 42
    assert snapshot["num_rounds"] == 3
    assert snapshot["synthesizer"] == "fed_hello"
    assert snapshot["data"]["dataset"] == "dummy.csv"


# --- experiment metadata in metrics ----------------------------------------


def test_metrics_files_written(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path,
        aggregated_metrics={"fidelity.mean_abs_diff": 0.5},
        centralized_metrics={"utility.tstr_auc": 0.9},
    )
    write_artifacts(ctx)  # type: ignore[arg-type]

    outdir = tmp_path / "test-run"
    fed = json.loads(outdir.joinpath("metrics.federated.json").read_text())
    assert fed["fidelity.mean_abs_diff"] == 0.5

    cent = json.loads(outdir.joinpath("metrics.centralized.json").read_text())
    assert cent["utility.tstr_auc"] == 0.9


def test_platform_metadata_in_metrics(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    write_artifacts(ctx)  # type: ignore[arg-type]

    outdir = tmp_path / "test-run"
    data = json.loads(outdir.joinpath("metadata.json").read_text())
    assert "platform.os" in data
    assert "platform.python_version" in data
    assert "platform.cpu_count" in data


def test_nan_metrics_written_as_null(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path,
        centralized_metrics={"utility.tstr_auc": float("nan")},
    )
    write_artifacts(ctx)  # type: ignore[arg-type]

    data = json.loads((tmp_path / "test-run" / "metrics.centralized.json").read_text())
    assert data["utility.tstr_auc"] is None


# --- synthetic.csv ---------------------------------------------------------


def test_synthetic_csv_written(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    write_artifacts(ctx)  # type: ignore[arg-type]

    df = pd.read_csv(tmp_path / "test-run" / "synthetic.csv")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


# --- output directory structure --------------------------------------------


def test_all_expected_files_present(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    write_artifacts(ctx)  # type: ignore[arg-type]

    outdir = tmp_path / "test-run"
    expected = {
        "config_snapshot.json",
        "metrics.federated.json",
        "metrics.centralized.json",
        "synthetic.csv",
    }
    actual = {f.name for f in outdir.iterdir()}
    assert expected.issubset(actual)
