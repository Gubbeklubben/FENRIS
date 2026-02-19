from pathlib import Path

from fedbench.config.builder import build_config


def test_minimal_config():
    cfg_dict = {
        "dataset": "datasets/breast_cancer.csv",
        "algorithm": "fed_noop",
        "partitioner": "iid-partitioner",
    }
    cfg = build_config(cfg_dict)
    assert cfg.data.dataset == str(Path(cfg_dict["dataset"]).resolve())
    assert cfg.data.partitioner == cfg_dict["partitioner"]
    assert cfg.algorithm == cfg_dict["algorithm"]