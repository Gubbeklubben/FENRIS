from collections import ChainMap
from dataclasses import fields, MISSING
from pathlib import Path
from typing import Any

from fedbench.config import DataConfig, MetricsConfig
from fedbench.config.config import Config
from fedbench.eval.evaluators import Category


def build_config(cli_input: dict[str, Any]) -> Config:
    dataconfig_defaults = build_defaults_dict(DataConfig)
    metricsconfig_defaults = build_defaults_dict(MetricsConfig)
    config_defaults = build_defaults_dict(Config)

    dataconfig_cli = build_cli_dict(DataConfig, cli_input)
    metricsconfig_cli = build_cli_dict(MetricsConfig, cli_input)
    config_cli = build_cli_dict(Config, cli_input)

    datacfg_combined = ChainMap(dataconfig_cli, dataconfig_defaults)
    metricscfg_combined = ChainMap(metricsconfig_cli, metricsconfig_defaults)
    cfg_combined = ChainMap(config_cli, config_defaults)

    path = Path(datacfg_combined["dataset"]).resolve()
    if path.is_dir():
        raise IsADirectoryError(f'"{path}" is a directory')
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} does not exist")
    datacfg_combined["dataset"] = str(path)

    # TODO: check partitioner

    if Category.UTILITY in metricscfg_combined["run_categories"] and datacfg_combined["target_col"] is None:
        raise ValueError("Target column must be specified when running utility metrics")

    for category in metricscfg_combined["run_categories"]:
        if category not in Category:
            raise ValueError(f"Category {category} is not supported")

    # TODO: check algorithm exists

    if cfg_combined["num_clients"] < 1:
        raise ValueError(f"Number of clients {cfg_combined["num_clients"]} is not supported")
    if cfg_combined["num_rounds"] < 1:
        raise ValueError(f"Number of rounds {cfg_combined["num_rounds"]} is not supported")
    if cfg_combined["test_size"] <= 0 or cfg_combined["test_size"] >= 1:
        raise ValueError(f"Test size {cfg_combined["test_size"]} is not supported, must be between 0 and 1")

    if not cfg_combined["outputdir"]:
        cfg_combined["outputdir"] = Path.cwd().joinpath("out")
    else:
        cfg_combined["outputdir"] = Path(cfg_combined["outputdir"]).resolve()

    if cfg_combined["num_synthetic_rows"] is not None and cfg_combined["num_synthetic_rows"] <= 0:
        raise ValueError(f"Number of synthetic rows {cfg_combined["num_synthetic_rows"]} is not supported")

    datacfg = DataConfig(**datacfg_combined)
    metricscfg = MetricsConfig(**metricscfg_combined)
    return Config(
        **cfg_combined,
        data=datacfg,
        metrics=metricscfg
    )



def build_defaults_dict(config_class):
    return {
        f.name: f.default
        for f in fields(config_class)
        if f.default is not MISSING
    }

def build_cli_dict(config_class, cli_input: dict[str, Any]) -> dict[str, Any]:

    cli_dict = {}
    for f in fields(config_class):
        if f.name in cli_input:
            cli_dict[f.name] = cli_input[f.name]
    return cli_dict