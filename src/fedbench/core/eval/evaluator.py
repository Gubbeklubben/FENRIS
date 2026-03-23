from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, StrEnum, auto
from typing import Any, Iterable, Literal

from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext


class Category(StrEnum):
    FIDELITY = "fidelity"
    UTILITY = "utility"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    SCALABILITY = "scalability"


class EvaluationMode(Flag):
    CENTRALIZED = auto()
    FEDERATED = auto()
    BOTH = CENTRALIZED | FEDERATED


@dataclass(frozen=True)
class MetricDescriptor:
    key: str
    default_stop_mode: Literal["min", "max"] | None = "min"
    suffix_type: Literal["sensitive", "target", None] = None


@dataclass(frozen=True)
class EvaluatorDescriptor:
    name: str
    category: Category
    eval_mode: EvaluationMode
    metrics: list[MetricDescriptor]


class Evaluator(ABC):
    @property
    @abstractmethod
    def metadata(self) -> EvaluatorDescriptor: ...

    # Centralized mode
    @abstractmethod
    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]: ...

    # Federated mode, client side
    @abstractmethod
    def local_evaluate(self, ctx: LocalEvalContext) -> Any: ...

    # Federated mode, server side
    @abstractmethod
    def aggregate(self, stats: Iterable[Any]) -> dict[str, float]: ...
