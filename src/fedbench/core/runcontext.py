from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from pandas import DataFrame

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import Partitioner, TableSchema
from fedbench.core.eval import EvaluationSuite
from fedbench.core.eventbus import EventBus
from fedbench.core.update import Update


@dataclass(frozen=True)
class Components:
    df_loader: Callable[[], DataFrame]
    algorithm: Algorithm
    partitioner: Partitioner
    eval_suite: EvaluationSuite


# The repetitive getter/setter's in RunContext is at least explicit,
# but could probably be implemented with python's descriptor protocol.
class RunContext:
    def __init__(self, run_id: str, config: Config, eventbus: EventBus) -> None:
        self._run_id = run_id
        self._config = config
        self._eventbus = eventbus
        self._components: Components | None = None
        self.df: DataFrame | None = None  # Less strict policy for now
        self._schema: TableSchema | None = None
        self._aggregated_state: Update | None = None
        self._aggregated_metrics: dict[str, float] | None = None
        # per client metrics?
        self._synthetic_df: DataFrame | None = None

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
    def components(self) -> Components:
        if self._components is None:
            raise RuntimeError("Property 'components' accessed before set.")
        return self._components

    @components.setter
    def components(self, components: Components) -> None:
        if self._components is not None:
            raise RuntimeError("Can only set components once.")
        self._components = components

    @property
    def schema(self) -> TableSchema:
        if self._schema is None:
            raise RuntimeError("Property 'schema' accessed before set.")
        return self._schema

    @schema.setter
    def schema(self, schema: TableSchema) -> None:
        if self._schema is not None:
            raise RuntimeError("Can only set 'schema' once.")
        self._schema = schema

    @property
    def aggregated_state(self) -> Update:
        if self._aggregated_state is None:
            raise RuntimeError("Property 'aggregated_state' accessed before set.")
        return self._aggregated_state

    @aggregated_state.setter
    def aggregated_state(self, state: Update) -> None:
        if self._aggregated_state is not None:
            raise RuntimeError("Can only set 'aggregated_state' once.")
        self._aggregated_state = state

    @property
    def aggregated_metrics(self) -> Mapping[str, float]:
        if self._aggregated_metrics is None:
            raise RuntimeError("Property 'aggregated_metrics' accessed before set.")
        return MappingProxyType(self._aggregated_metrics)

    @aggregated_metrics.setter
    def aggregated_metrics(self, metrics: dict[str, float]) -> None:
        if self._aggregated_metrics is not None:
            raise RuntimeError("Can only set 'aggregated_metrics' once.")
        self._aggregated_metrics = metrics

    @property
    def synthetic_df(self) -> DataFrame:
        if self._synthetic_df is None:
            raise RuntimeError("Property 'synthetic_df' accessed before set.")
        return self._synthetic_df

    @synthetic_df.setter
    def synthetic_df(self, df: DataFrame) -> None:
        if self._synthetic_df is not None:
            raise RuntimeError("Can only set 'synthetic_df' once.")

        if not isinstance(df, DataFrame):
            raise ValueError(f"Expected a DataFrame, got {type(df)}.")

        self._synthetic_df = df
