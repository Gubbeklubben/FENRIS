import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, StrEnum, auto
from typing import Any, Iterable, Literal

from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z_]+", "_", text.lower())


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

    def get_metric_descriptor_dict(
        self,
        target_column: str | None = None,
        sensitive_columns: tuple[str, ...] | None = None,
    ) -> dict[str, MetricDescriptor]:
        descriptors: dict[str, MetricDescriptor] = {}
        for metric in self.metadata.metrics:
            if sensitive_columns and metric.suffix_type == "sensitive":
                for suffix in sensitive_columns:
                    descriptors[f"{metric.key}.{normalize_key(suffix)}"] = metric
            elif target_column and metric.suffix_type == "target":
                descriptors[f"{metric.key}.{normalize_key(target_column)}"] = metric
            else:
                descriptors[metric.key] = metric
        return descriptors

    def get_metric_keys(
        self,
        target_column: str | None = None,
        sensitive_columns: tuple[str, ...] | None = None,
    ) -> Iterable[str]:
        return self.get_metric_descriptor_dict(target_column, sensitive_columns).keys()

    def _nan_result(self) -> dict[str, float]:
        return {key: math.nan for key in self.get_metric_keys()}

    # Centralized mode
    @abstractmethod
    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]: ...

    # Federated mode, client side
    @abstractmethod
    def local_evaluate(self, ctx: LocalEvalContext) -> Any: ...

    # Federated mode, server side
    @abstractmethod
    def aggregate(self, stats: Iterable[Any]) -> dict[str, float]: ...
