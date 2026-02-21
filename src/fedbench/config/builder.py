from collections import ChainMap
from dataclasses import fields, MISSING
from pathlib import Path
from typing import Any, Mapping

from fedbench.config import DataConfig, MetricsConfig
from fedbench.config.config import Config, ConfigCls
from fedbench.eval.evaluators import Category
from fedbench.data.partitioners import registry as partitioner_reg
from fedbench.algorithms import registry as algorithm_reg


def build_config(cli_input: dict[str, Any]) -> Config:
    datacfg_defaults = build_defaults_dict(DataConfig)
    metricscfg_defaults = build_defaults_dict(MetricsConfig)
    cfg_defaults = build_defaults_dict(Config)

    datacfg_cli = build_cli_dict(DataConfig, cli_input)
    metricscfg_cli = build_cli_dict(MetricsConfig, cli_input)
    cfg_cli = build_cli_dict(Config, cli_input)

    datacfg_combined = ChainMap(datacfg_cli, datacfg_defaults)
    metricscfg_combined = ChainMap(metricscfg_cli, metricscfg_defaults)
    cfg_combined = ChainMap(cfg_cli, cfg_defaults)

    datacfg_combined["dataset"] = str(Path(datacfg_combined["dataset"]).resolve())

    if not cfg_combined["outputdir"]:
        cfg_combined["outputdir"] = str(Path.cwd().joinpath("out"))
    else:
        cfg_combined["outputdir"] = str(Path(cfg_combined["outputdir"]).resolve())

    validate_all_configs(
        cfg_combined,
        datacfg_combined,
        metricscfg_combined,
    )

    partitioner = partitioner_reg.call(
        datacfg_combined["partitioner"],
        **datacfg_combined["partitioner_kwargs"]
    )
    cfg_combined["num_clients"] = partitioner.num_partitions

    cfg_combined["data"] = DataConfig(**datacfg_combined)
    cfg_combined["metrics"] = MetricsConfig(**metricscfg_combined)

    return Config(**cfg_combined)


def validate_data_config(data_config: Mapping[str, Any]) -> None:
    path = Path(data_config["dataset"])
    if path.is_dir():
        raise IsADirectoryError(f'"{path}" is a directory')
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} does not exist")

    if data_config["partitioner"] not in partitioner_reg:
        raise ValueError(f"Partitioner {data_config['partitioner']} is not registered")


def validate_metrics_config(metrics_config: Mapping[str, Any]) -> None:
    for category in metrics_config["run_categories"]:
        if category not in Category:
            raise ValueError(f"Category {category} is not supported")


def validate_config(config: Mapping[str, Any]) -> None:
    if config["algorithm"] not in algorithm_reg:
        raise ValueError(f"Algorithm {config["algorithm"]} is not registered")

    if config["num_rounds"] < 1:
        raise ValueError(f"Number of rounds {config["num_rounds"]} is not supported")
    if config["test_size"] <= 0 or config["test_size"] >= 1:
        raise ValueError(f"Test size {config["test_size"]} is not supported, must be between 0 and 1")

    if config["num_synthetic_rows"] is not None and config["num_synthetic_rows"] <= 0:
        raise ValueError(f"Number of synthetic rows {config["num_synthetic_rows"]} is not supported")


def validate_all_configs(
        config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        metrics_config: Mapping[str, Any],
) -> None:
    validate_data_config(data_config)
    validate_metrics_config(metrics_config)
    validate_config(config)

    if Category.UTILITY in metrics_config["run_categories"] and data_config["target_col"] is None:
        raise ValueError("Target column must be specified when running utility metrics")


def build_defaults_dict(config_cls: ConfigCls) -> dict[str, Any]:
    defaults_dict = {}
    for f in fields(config_cls):
        if f.default is not MISSING:
            defaults_dict[f.name] = f.default
        elif f.default_factory is not MISSING:
            defaults_dict[f.name] = f.default_factory()
    return defaults_dict


def build_cli_dict(config_cls: ConfigCls, cli_input: dict[str, Any]) -> dict[str, Any]:
    cli_dict = {}
    for f in fields(config_cls):
        if f.name in cli_input:
            cli_dict[f.name] = cli_input[f.name]
    return cli_dict