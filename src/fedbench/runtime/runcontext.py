from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, cast, overload

from pandas import DataFrame

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm, Coordinator, GlobalInitArtifacts
from fedbench.core.data import PartitionedDataset, Partitioner
from fedbench.core.eval import EvaluationSuite
from fedbench.core.payload import Payload
from fedbench.runtime.eventbus import EventBus
from fedbench.runtime.scalability_collector import ScalabilityCollector


class _RunCtxField[T]:
    """Descriptor encapsulating get/set logic for RunContext fields.

    Must always be declared as a class attribute in the class body,
    never be attached to a class dynamically.

    Semantics:
    ---------
    - Get before set -> AttributeError.
    - Set more than once -> RuntimeError.

    Deliberately assumes runtime checking of types is someone else's
    responsibility and does no such thing.
    """

    @property
    def name(self) -> str:
        return self._name.lstrip("_")

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._name = f"_{name}"

    @overload
    def __get__(self, instance: None, owner: type[object]) -> _RunCtxField[T]: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> T: ...

    def __get__(
        self, instance: object | None, owner: type[object]
    ) -> T | _RunCtxField[T]:

        if instance is None:
            return self

        if not hasattr(instance, self._name):
            raise AttributeError(f"{instance}: '{self.name}' accessed before set.")
        # noinspection PyUnnecessaryCast
        return cast(T, getattr(instance, self._name))

    def __set__(self, instance: object, value: T) -> None:
        if hasattr(instance, self._name):
            raise RuntimeError(f"{instance}: '{self.name}' already set.")

        setattr(instance, self._name, value)


class RunContext:
    # fmt: off
    algorithm             = _RunCtxField[Algorithm]()
    coordinator           = _RunCtxField[Coordinator]()
    df_loader             = _RunCtxField[Callable[[], DataFrame]]()
    partitioner           = _RunCtxField[Partitioner]()
    eval_suite            = _RunCtxField[EvaluationSuite]()
    dataset               = _RunCtxField[PartitionedDataset]()
    global_init_artifacts = _RunCtxField[GlobalInitArtifacts]()
    train_artifacts      = _RunCtxField[Payload]()
    per_client_metrics    = _RunCtxField[Mapping[int, Mapping[str, Any]]]()
    aggregated_metrics    = _RunCtxField[Mapping[str, float]]()
    centralized_metrics   = _RunCtxField[Mapping[str, float]]()
    synthetic_df          = _RunCtxField[DataFrame]()
    # fmt: on

    def __init__(
        self,
        run_id: str,
        config: Config,
        eventbus: EventBus,
        scalability_collector: ScalabilityCollector,
    ) -> None:

        self._run_id = run_id
        self._config = config
        self._eventbus = eventbus
        self._scalability_collector = scalability_collector

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def config(self) -> Config:
        return self._config

    @property
    def eventbus(self) -> EventBus:
        return self._eventbus

    @property
    def scalability_collector(self) -> ScalabilityCollector:
        return self._scalability_collector
