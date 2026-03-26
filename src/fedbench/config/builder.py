import csv
import inspect
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from fedbench.config.config import (
    Config,
    ConfigCls,
    DataConfig,
    MetricsConfig,
    SeedConfig,
)
from fedbench.config.parsing import parse_for_function
from fedbench.core.algorithm import Synthesizer
from fedbench.core.data import Partitioner
from fedbench.core.eval import Category, EvaluationSuite
from fedbench.core.eval.evaluator import EvaluationMode
from fedbench.runtime.registry import FactoryRegistry
from fedbench.runtime.registry_builder import build_evaluator_registry


def build_config(
    cli_input: dict[str, Any],
    synthesizer_registry: FactoryRegistry[Synthesizer],
    partitioner_registry: FactoryRegistry[Partitioner],
) -> Config:
    # Remove seed from cli_input so it doesn't become part of the cfg dict.
    # We use cli_input["seed"] to construct Config.seed of type SeedConfig.
    default_seed = inspect.signature(SeedConfig.from_master).parameters["seed"].default
    seed = cli_input.pop("seed", default_seed)
    seed_cfg = SeedConfig.from_master(seed)

    # Build dicts containing only kv pairs relevant for the associated cfg
    data_cfg = build_cli_dict(DataConfig, cli_input)
    metrics_cfg = build_cli_dict(MetricsConfig, cli_input)
    cfg = build_cli_dict(Config, cli_input)

    resolve_dataset_path(data_cfg)
    validate_column_names(data_cfg)
    resolve_outputdir(cfg)
    resolve_run_categories(metrics_cfg)
    validate_stop_metrics(metrics_cfg, data_cfg)

    parse_synthesizer_kwargs(cfg, synthesizer_registry)
    parse_partitioner_kwargs(cfg, data_cfg, seed_cfg, partitioner_registry)

    # Build complete config object
    return Config(
        **cfg,
        data=DataConfig(**data_cfg),
        metrics=MetricsConfig(**metrics_cfg),
        seed=seed_cfg,
    )


def build_cli_dict(config_cls: ConfigCls, cli_input: dict[str, Any]) -> dict[str, Any]:
    return {
        f.name: cli_input[f.name]  # nofmt
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


def validate_column_names(data_cfg: dict[str, Any]) -> None:
    """Fail fast if target_col or sensitive_cols reference non-existent columns."""
    with open(data_cfg["dataset"]) as f:
        header = next(csv.reader(f))

    target = data_cfg.get("target_col")
    if target is not None and target not in header:
        raise ValueError(
            f"--target-col '{target}' not found in dataset. Available columns: {header}"
        )

    for col in data_cfg.get("sensitive_cols", ()):
        if col not in header:
            raise ValueError(
                f"--sensitive-cols: '{col}' not found in dataset. "
                f"Available columns: {header}"
            )


def resolve_outputdir(cfg: dict[str, Any]) -> None:
    # Resolve canonical outputdir, or use ./out if none specified
    if not cfg.get("outputdir"):
        path = Path.cwd().joinpath("out")
    else:
        path = Path(cfg["outputdir"])
    cfg["outputdir"] = str(Path(path).expanduser().resolve())


def resolve_run_categories(metrics_cfg: dict[str, Any]) -> None:
    # Map metrics categories from strings to enum members;
    # default to all categories if omitted.
    if not metrics_cfg.get("run_categories"):
        metrics_cfg["run_categories"] = tuple(Category)
    else:
        metrics_cfg["run_categories"] = tuple(
            Category(v) for v in metrics_cfg["run_categories"]
        )


def validate_stop_metrics(
    metrics_cfg: dict[str, Any], data_cfg: dict[str, Any]
) -> None:
    if not metrics_cfg.get("early_stop"):
        return
    if "stop_metric" not in metrics_cfg:
        raise ValueError("stop_metric must be specified when early_stop is enabled")

    eval_suite = EvaluationSuite.with_evaluator_categories(
        build_evaluator_registry(),
        metrics_cfg["run_categories"],
    )

    try:
        evaluator, metric = eval_suite.get_evaluator_for_metric_key(
            metrics_cfg["stop_metric"],
            data_cfg["target_col"],
            data_cfg["sensitive_cols"],
        )
    except KeyError as e:
        raise ValueError(
            f"Specified stop metric `{metrics_cfg['stop_metric']}` "
            f"is not emitted by any evaluator in the current evaluation suite."
        ) from e

    if EvaluationMode.CENTRALIZED not in evaluator.metadata.eval_mode:
        raise ValueError(
            f"Metric `{metrics_cfg['stop_metric']}` does not support "
            f"centralized evaluation and cannot be used as a stop metric"
        )
    if "stop_mode" not in metrics_cfg:
        metrics_cfg["stop_mode"] = metric.default_stop_mode


def parse_synthesizer_kwargs(
    cfg: dict[str, Any], synthesizer_registry: FactoryRegistry[Synthesizer]
) -> None:

    name = cfg["synthesizer"]
    # Check that specified synthesizer is registered
    if name not in synthesizer_registry:
        raise ValueError(f"Synthesizer {name} is not registered")

    # Parse kwargs and map to correct types
    cfg["synthesizer_kwargs"] = parse_for_function(
        synthesizer_registry.load(name),
        cfg.get("synthesizer_kwargs", {}),
    )


def _get_config_value_or_default(
    cfg: dict[str, Any],
    param: str,
) -> Any:
    default = next(f.default for f in fields(Config) if f.name == param)
    return cfg.get(param, default)


def _reject_and_inject(
    name: str,
    value: Any,
    raw_kwargs: dict[str, Any],
    factory_params: Mapping[str, Any],
    expected_type: type,
    *,
    required: bool = False,
) -> None:
    """Reject user-specified framework param; inject into raw kwargs before parse."""
    if name in raw_kwargs:
        raise ValueError(
            f"'{name}' must not be specified in --partitioner-kwargs. "
            "It is controlled by the framework."
        )
    if name not in factory_params:
        if required:
            raise ValueError(f"Partitioner must accept a '{name}' parameter.")
        return
    if factory_params[name].annotation is not expected_type:
        raise TypeError(
            f"Partitioner parameter '{name}' must have type {expected_type}."
        )
    raw_kwargs[name] = value


def parse_partitioner_kwargs(
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    seed: SeedConfig,
    partitioner_registry: FactoryRegistry[Partitioner],
) -> None:
    # Check that specified partitioner is registered
    if data_cfg["partitioner"] not in partitioner_registry:
        raise ValueError(f"Partitioner {data_cfg['partitioner']} is not registered")

    partitioner_kwargs = data_cfg.get("partitioner_kwargs", {})
    partitioner_factory = partitioner_registry.load(data_cfg["partitioner"])
    params = inspect.signature(partitioner_factory).parameters

    # Inject framework-controlled parameters (reject if user specified).
    num_partitions = _get_config_value_or_default(cfg, param="num_clients")
    _reject_and_inject(
        name="num_partitions",
        value=num_partitions,
        raw_kwargs=partitioner_kwargs,
        factory_params=params,
        expected_type=int,
        required=True,
    )

    _reject_and_inject(
        name="seed",
        value=seed.partitioning,
        raw_kwargs=partitioner_kwargs,
        factory_params=params,
        expected_type=int,
    )

    # Validate and parse all kwargs (user + framework injected).
    data_cfg["partitioner_kwargs"] = parse_for_function(
        partitioner_factory,
        partitioner_kwargs,
    )
