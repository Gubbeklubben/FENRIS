import pytest
from pathlib import Path

from fedbench.config.builder import build_config
from fedbench.eval.evaluators import Category


# --- helpers ---------------------------------------------------------------

def minimal_valid_cfg(tmp_path: Path, **overrides):
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n")

    base = {
        "dataset": str(dataset),
        "algorithm": "fed_noop",
        "partitioner": "iid-partitioner",
    }
    base.update(overrides)
    return base

# --- minimal config builder validation ------------------------------------

def test_minimal_config(tmp_path):
    cfg_dict = minimal_valid_cfg(tmp_path)

    cfg = build_config(cfg_dict)

    assert cfg.data.dataset == str(Path(cfg_dict["dataset"]).resolve())
    assert cfg.data.partitioner == cfg_dict["partitioner"]
    assert cfg.algorithm == cfg_dict["algorithm"]


# --- dataset path validation ----------------------------------------------

def test_dataset_is_directory_raises(tmp_path):
    cfg = minimal_valid_cfg(tmp_path)
    cfg["dataset"] = str(tmp_path)

    with pytest.raises(IsADirectoryError):
        build_config(cfg)


def test_dataset_does_not_exist_raises(tmp_path):
    cfg = minimal_valid_cfg(tmp_path)
    cfg["dataset"] = str(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError):
        build_config(cfg)


# --- metrics / category validation ----------------------------------------

def test_utility_category_without_target_col_raises(tmp_path):
    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=(Category.UTILITY,),
        target_col=None,
    )

    with pytest.raises(ValueError, match="Target column must be specified"):
        build_config(cfg)


def test_unsupported_category_raises(tmp_path):
    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=("not-a-category",),
        target_col="y",
    )

    with pytest.raises(ValueError, match="Category .* is not supported"):
        build_config(cfg)


# --- numeric config validation --------------------------------------------

@pytest.mark.parametrize("num_clients", [0, -1])
def test_invalid_num_clients_raises(tmp_path, num_clients):
    cfg = minimal_valid_cfg(tmp_path, num_clients=num_clients)

    with pytest.raises(ValueError, match="Number of clients"):
        build_config(cfg)


@pytest.mark.parametrize("num_rounds", [0, -5])
def test_invalid_num_rounds_raises(tmp_path, num_rounds):
    cfg = minimal_valid_cfg(tmp_path, num_rounds=num_rounds)

    with pytest.raises(ValueError, match="Number of rounds"):
        build_config(cfg)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5])
def test_invalid_test_size_raises(tmp_path, test_size):
    cfg = minimal_valid_cfg(tmp_path, test_size=test_size)

    with pytest.raises(ValueError, match="Test size"):
        build_config(cfg)


def test_invalid_num_synthetic_rows_raises(tmp_path):
    cfg = minimal_valid_cfg(tmp_path, num_synthetic_rows=0)

    with pytest.raises(ValueError, match="Number of synthetic rows"):
        build_config(cfg)


# --- outputdir behavior ----------------------------------------------------

def test_default_outputdir_is_cwd_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = minimal_valid_cfg(tmp_path)

    config = build_config(cfg)

    assert config.outputdir == tmp_path / "out"


def test_custom_outputdir_is_resolved(tmp_path):
    out = tmp_path / "results"
    cfg = minimal_valid_cfg(tmp_path, outputdir=str(out))

    config = build_config(cfg)

    assert config.outputdir == out.resolve()


# --- positive sanity checks ------------------------------------------------

def test_valid_utility_category_with_target_col(tmp_path):
    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=(Category.UTILITY,),
        target_col="label",
    )

    config = build_config(cfg)

    assert config.data.target_col == "label"
    assert Category.UTILITY in config.metrics.run_categories
