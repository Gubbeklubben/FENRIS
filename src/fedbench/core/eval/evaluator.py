from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Iterable

from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext


class Category(StrEnum):
    FIDELITY = "fidelity"
    UTILITY = "utility"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    SCALABILITY = "scalability"


class Evaluator(ABC):
    # Centralized mode
    @abstractmethod
    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]: ...

    # Federated mode, client side
    @abstractmethod
    def local_evaluate(self, ctx: LocalEvalContext) -> Any: ...

    # Federated mode, server side
    @abstractmethod
    def aggregate(self, stats: Iterable[Any]) -> dict[str, float]: ...
