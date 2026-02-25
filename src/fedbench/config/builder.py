import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any

from fedbench.util.parsing import parse_for_function
from fedbench.config.config import DataConfig, MetricsConfig, Config, ConfigCls
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import Partitioner
from fedbench.core.eval import Category
from fedbench.core.factory_registry import FactoryRegistry


def build_config(
        cli_input: dict[str, Any],
        algorithm_registry: FactoryRegistry[Algorithm],
        partitioner_registry: FactoryRegistry[Partitioner]) -> Config:
    # Build dicts containing only kv pairs relevant for the associated cfg
    data_cfg = build_cli_dict(DataConfig, cli_input)
    metrics_cfg = build_cli_dict(MetricsConfig, cli_input)
    cfg = build_cli_dict(Config, cli_input)

    resolve_dataset_path(data_cfg)
    resolve_outputdir(cfg)
    resolve_run_categories(metrics_cfg)

    parse_algorithm_kwargs(cfg, algorithm_registry)
    parse_partitioner_kwargs(data_cfg, cfg, partitioner_registry)

    # Build complete config object
    cfg["data"] = DataConfig(**data_cfg)
    cfg["metrics"] = MetricsConfig(**metrics_cfg)
    return Config(**cfg)


def build_cli_dict(config_cls: ConfigCls, cli_input: dict[str, Any]) -> dict[str, Any]:
    return {
        f.name: cli_input[f.name]
        for f in fields(config_cls)
        if f.name in cli_input
    }


def resolve_dataset_path(data_cfg: dict[str, Any]) -> None:
    # Resolve canonical dataset path
    path = Path(data_cfg["dataset"]).expanduser().resolve()
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory")
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} does not exist")
    data_cfg["dataset"] = str(path)


def resolve_outputdir(cfg: dict[str, Any]) -> None:
    # Resolve canonical outputdir, or use ./out if none specified
    if not cfg.get("outputdir"):
        path = Path.cwd().joinpath("out")
    else:
        path = Path(cfg["outputdir"])
    cfg["outputdir"] = str(Path(path).expanduser().resolve())


def resolve_run_categories(metrics_cfg: dict[str, Any]) -> None:
    # Map metrics categories from strings to enum members; default to all metrics categories if omitted
    if not metrics_cfg.get("run_categories"):
        metrics_cfg["run_categories"] = tuple(Category)
    else:
        metrics_cfg["run_categories"] = tuple(Category(v) for v in metrics_cfg["run_categories"])


def parse_algorithm_kwargs(cfg: dict[str, Any], algorithm_registry: FactoryRegistry[Algorithm]) -> None:
    # Check that specified algorithm is registered
    if cfg["algorithm"] not in algorithm_registry:
        raise ValueError(f"Algorithm {cfg["algorithm"]} is not registered")

    # Parse algorithm kwargs and map to correct types
    cfg["algorithm_kwargs"] = parse_for_function(
        algorithm_registry.load(cfg["algorithm"]),
        cfg.get("algorithm_kwargs", {})
    )


def parse_partitioner_kwargs(data_cfg: dict[str, Any], cfg: dict[str, Any], partitioner_registry: FactoryRegistry[Partitioner]) -> None:
    # Check that specified partitioner is registered
    if data_cfg["partitioner"] not in partitioner_registry:
        raise ValueError(f"Partitioner {data_cfg["partitioner"]} is not registered")

    # Check that user has not specified num_partitions explicitly
    partitioner_kwargs = data_cfg.get("partitioner_kwargs", {})
    if "num_partitions" in partitioner_kwargs:
        raise ValueError("num_partitions should not be explicitly specified in partitioner kwargs. "
                         "It is determined automatically based on the num_clients CLI parameter.")

    # Check that the partitioner factory takes a num_partitions parameter of type int
    partitioner_factory = partitioner_registry.load(data_cfg["partitioner"])
    params = inspect.signature(partitioner_factory).parameters
    if not "num_partitions" in params.keys():
        raise ValueError(f"Partitioner factory {data_cfg["partitioner"]} must have a num_partitions parameter")
    if not params["num_partitions"].annotation == int:
        raise TypeError(f"Partitioner factory {data_cfg["partitioner"]} must have a num_partitions parameter of type int")

    # Parse partitioner kwargs and map to correct types, then add num_partitions based on num_clients
    default_num_clients = next(f.default for f in fields(Config) if f.name == "num_clients")
    partitioner_kwargs["num_partitions"] = cfg.get("num_clients", default_num_clients)
    data_cfg["partitioner_kwargs"] = parse_for_function(
        partitioner_factory,
        partitioner_kwargs
    )