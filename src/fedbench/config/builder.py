import csv
import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, TypeVar

from fedbench.config.config import (
    Config,
    ConfigCls,
    DataConfig,
    MetricsConfig,
    SeedConfig,
)
from fedbench.config.parsing import parse_kwargs_for_function
from fedbench.core.algorithm import Coordinator, Synthesizer
from fedbench.core.component import Component
from fedbench.core.data import Partitioner
from fedbench.core.eval import Category
from fedbench.core.eval.evaluator import EvaluationMode
from fedbench.runtime.factory import create_evaluation_suite
from fedbench.runtime.registry import Group, Registry

ComponentT = TypeVar("ComponentT", bound=Component)


def build_config(
    cli_input: dict[str, Any],
    synthesizer_registry: Registry | None = None,
    coordinator_registry: Registry | None = None,
    partitioner_registry: Registry | None = None,
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
    resolve_schema_path(data_cfg)
    validate_column_names(data_cfg)
    resolve_outputdir(cfg)
    resolve_run_categories(metrics_cfg)
    validate_stop_metrics(metrics_cfg, data_cfg)

    synthesizer_registry = synthesizer_registry or Group.SYNTHESIZERS.get_registry()
    coordinator_registry = coordinator_registry or Group.COORDINATORS.get_registry()
    partitioner_registry = partitioner_registry or Group.PARTITIONERS.get_registry()

    validate_synthesizer(synthesizer_registry, cfg)
    validate_coordinator(coordinator_registry, cfg)
    validate_partitioner(partitioner_registry, cfg, data_cfg, seed_cfg)

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
        raise IsADirectoryError(f"`{path}` is a directory")
    if not path.exists():
        raise FileNotFoundError(f"Dataset file `{path}` does not exist")
    data_cfg["dataset"] = str(path)


def resolve_schema_path(data_cfg: dict[str, Any]) -> None:
    if data_cfg.get("schema"):
        path = Path(data_cfg["schema"]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Schema file `{path}` does not exist")
    else:
        path = Path(data_cfg["dataset"]).with_suffix(".schema.json")
        if data_cfg.get("generate_input_schema") and path.exists():
            raise FileExistsError(
                f"Cannot generate input schema file. "
                f"`{path}` already exists and would be overwritten."
            )
        # If we're not outputting an input schema file, we don't need to check whether
        # file exists. Schema will then be auto-inferred during loading (and possibly
        # written back as an input schema later) if the default file is not found.

    if path.is_dir():
        raise IsADirectoryError(f"`{path}` is a directory")

    data_cfg["schema"] = str(path)


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
        metrics_cfg["run_categories"] = (
            Category.SCALABILITY,
            *(Category(v) for v in metrics_cfg["run_categories"]),
        )


def validate_stop_metrics(
    metrics_cfg: dict[str, Any], data_cfg: dict[str, Any]
) -> None:
    if not metrics_cfg.get("early_stop"):
        return
    if not metrics_cfg.get("stop_metric"):
        raise ValueError("stop_metric must be specified when early_stop is enabled")

    eval_suite = create_evaluation_suite(metrics_cfg["run_categories"])
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
    if not metrics_cfg.get("stop_mode"):
        metrics_cfg["stop_mode"] = metric.default_stop_mode


def validate_synthesizer(registry: Registry, cfg: dict[str, Any]) -> None:

    def callback(factory: type[Synthesizer]) -> None:
        if not hasattr(factory, "SUPPORTS_COORDINATORS"):
            raise AttributeError(
                f"Synthesizer `{cfg['synthesizer']}` does not declare supported "
                f"coordinators via the SUPPORTS_COORDINATORS attribute."
            )
        if cfg["coordinator"] not in factory.SUPPORTS_COORDINATORS:
            raise ValueError(
                f"Synthesizer {cfg['synthesizer']} does not support "
                f"coordinator {cfg['coordinator']}. "
                f"Supported coordinators: "
                f"{
                    None
                    if not factory.SUPPORTS_COORDINATORS
                    else ', '.join(factory.SUPPORTS_COORDINATORS)
                }."
            )

    validate_component(Synthesizer, registry, cfg, callback)  # type: ignore[type-abstract]


def validate_coordinator(registry: Registry, cfg: dict[str, Any]) -> None:
    validate_component(Coordinator, registry, cfg)  # type: ignore[type-abstract]


def validate_partitioner(
    registry: Registry, cfg: dict[str, Any], data_cfg: dict[str, Any], seed: SeedConfig
) -> None:

    def callback(factory: type[Partitioner]) -> None:
        inject_partitioner_kwargs(factory, cfg, data_cfg, seed)

    validate_component(Partitioner, registry, data_cfg, callback)  # type: ignore[type-abstract]


def validate_component(
    component_type: type[ComponentT],
    registry: Registry,
    cfg: dict[str, Any],
    preprocess_callback: Callable[[type[ComponentT]], None] | None = None,
) -> None:
    _component_type = component_type.__name__.lower()
    # Check that specified component is registered in the associated registry
    if cfg[_component_type] not in registry:
        raise ValueError(
            f"`{cfg[_component_type]}` is not a registered {_component_type}."
        )

    # Ensure kwargs dict exists
    kwargs_key = f"{_component_type}_kwargs"
    cfg.setdefault(kwargs_key, {})

    # Do component specific preprocessing like injecting partitioner kwargs
    factory = registry.load(cfg[_component_type])
    if preprocess_callback:
        preprocess_callback(factory)

    # Parse kwargs from string values into their correct types
    cfg[kwargs_key] = parse_kwargs_for_function(factory, cfg[kwargs_key])


def inject_partitioner_kwargs(
    factory: Callable[..., Any],
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    seed: SeedConfig,
) -> None:
    params = inspect.signature(factory).parameters
    if "num_partitions" not in params:
        raise ValueError(
            f"{factory.__name__} must accept a `num_partitions` parameter."
        )

    injected_kwargs = {"num_partitions": cfg.get("num_clients", Config.num_clients)}
    if "seed" in params:
        injected_kwargs["seed"] = seed.partitioning

    # Reject manually specified kwargs that should be framework-controlled
    if rejected := set(data_cfg["partitioner_kwargs"]) & set(injected_kwargs.keys()):
        raise ValueError(
            f"The following parameters cannot be specified as kwargs for "
            f"{factory.__name__}: {rejected}. "
            f"They are injected directly by the framework."
        )

    data_cfg["partitioner_kwargs"].update(injected_kwargs)
