from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fedbench.config.config import Config, DataConfig
from fedbench.core.update import Update
from fedbench.runtime.pipeline import _save_model_weights, write_artifacts

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

_requires_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")


def _make_config(tmp_path: Path) -> Config:
    return Config(
        algorithm="fed_hello",
        data=DataConfig(dataset="dummy.csv", partitioner="iid-partitioner"),
        outputdir=str(tmp_path),
        seed=42,
        num_rounds=3,
        num_clients=3,
    )


def _make_ctx(
    tmp_path: Path,
    *,
    aggregated_metrics: dict[str, float] | None = None,
    centralized_metrics: dict[str, float] | None = None,
    aggregated_state: Update | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=_make_config(tmp_path),
        run_id="test-run",
        aggregated_metrics=aggregated_metrics or {},
        centralized_metrics=centralized_metrics or {},
        synthetic_df=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        aggregated_state=aggregated_state or Update(),
    )


# --- config_snapshot.json --------------------------------------------------


def test_config_snapshot_written(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    write_artifacts(ctx)  # type: ignore[arg-type]

    snapshot = json.loads(
        (tmp_path / "test-run" / "config_snapshot.json").read_text()
    )
    assert snapshot["seed"] == 42
    assert snapshot["num_rounds"] == 3
    assert snapshot["algorithm"] == "fed_hello"
    assert snapshot["data"]["dataset"] == "dummy.csv"


# --- experiment metadata in metrics ----------------------------------------


def test_experiment_metadata_in_metrics(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path,
        aggregated_metrics={"fidelity.mean_abs_diff": 0.5},
        centralized_metrics={"utility.tstr_auc": 0.9},
    )
    write_artifacts(ctx)  # type: ignore[arg-type]

    outdir = tmp_path / "test-run"
    for name in ("federated", "centralized"):
        data = json.loads(outdir.joinpath(f"metrics.{name}.json").read_text())
        assert data["experiment.seed"] == 42
        assert data["experiment.num_rounds"] == 3
        assert data["experiment.num_clients"] == 3
        assert data["experiment.generator_type"] == "fed_hello"
        assert data["experiment.metric_focus"] is None
        assert data["experiment.aggregation_variant"] is None
        assert data["experiment.model_scope"] is None

    fed = json.loads(outdir.joinpath("metrics.federated.json").read_text())
    assert fed["fidelity.mean_abs_diff"] == 0.5

    cent = json.loads(outdir.joinpath("metrics.centralized.json").read_text())
    assert cent["utility.tstr_auc"] == 0.9


def test_nan_metrics_written_as_null(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path,
        centralized_metrics={"utility.tstr_auc": float("nan")},
    )
    write_artifacts(ctx)  # type: ignore[arg-type]

    data = json.loads(
        (tmp_path / "test-run" / "metrics.centralized.json").read_text()
    )
    assert data["utility.tstr_auc"] is None


# --- synthetic.csv ---------------------------------------------------------


def test_synthetic_csv_written(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    write_artifacts(ctx)  # type: ignore[arg-type]

    df = pd.read_csv(tmp_path / "test-run" / "synthetic.csv")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


# --- model weight saving ---------------------------------------------------


@_requires_torch
def test_save_synthesizer_weights(tmp_path: Path) -> None:
    state_dict = {"layer.weight": torch.tensor([1.0, 2.0])}
    state = Update(arrays={"state": state_dict})

    _save_model_weights(tmp_path, state)

    pt_file = tmp_path / "final_synthesizer.pt"
    assert pt_file.exists()
    loaded = torch.load(pt_file, weights_only=True)
    assert torch.equal(loaded["layer.weight"], torch.tensor([1.0, 2.0]))


def test_no_weights_no_pt_files(tmp_path: Path) -> None:
    state = Update()
    _save_model_weights(tmp_path, state)

    assert not (tmp_path / "final_synthesizer.pt").exists()


def test_ndarray_state_not_saved_as_pt(tmp_path: Path) -> None:
    """list[NDArray] values should not be torch.saved."""
    state = Update(arrays={"state": [np.array([1.0, 2.0])]})
    _save_model_weights(tmp_path, state)

    assert not (tmp_path / "final_synthesizer.pt").exists()


@_requires_torch
def test_save_weights_no_torch(tmp_path: Path) -> None:
    """When torch is not importable, _save_model_weights is a no-op."""
    state_dict = {"layer.weight": torch.tensor([1.0])}
    state = Update(arrays={"state": state_dict})

    with patch.dict("sys.modules", {"torch": None}):
        _save_model_weights(tmp_path, state)

    assert not (tmp_path / "final_synthesizer.pt").exists()


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
