from collections.abc import Callable
from dataclasses import dataclass

from pandas import DataFrame

from fedbench.config import Config
from fedbench.core.algorithm import Algorithm
from fedbench.core.data import TableSchema, Partitioner
from fedbench.core.eval import EvaluationSuite
from fedbench.core.eventbus import EventBus
from fedbench.core.update import Update


@dataclass(frozen=True)
class Components:
    df_loader: Callable[[], tuple[DataFrame, TableSchema]]
    algorithm: Algorithm
    partitioner: Partitioner
    eval_suite: EvaluationSuite


class RunContext:
    def __init__(self, run_id: str, config: Config, eventbus: EventBus) -> None:
        self._run_id = run_id
        self._config = config
        self._eventbus = eventbus
        self._components: Components | None = None
        self._final_aggregated_state: Update | None = None
        # per client metrics?
        # aggregated training metrics?
        # globally created synthetic data
        # in memory repr of final metrics output

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
            raise ValueError("Property components is not set.")
        return self._components

    @components.setter
    def components(self, components: Components) -> None:
        if self._components is not None:
            raise ValueError("Can only set components once.")
        self._components = components

    @property
    def final_aggregated_state(self) -> Update | None:
        return self._final_aggregated_state

    @final_aggregated_state.setter
    def final_aggregated_state(self, final_aggregated_state: Update) -> None:
        if self._final_aggregated_state is not None:
            raise ValueError("Can only set final_aggregated_state once.")
        self._final_aggregated_state = final_aggregated_state
