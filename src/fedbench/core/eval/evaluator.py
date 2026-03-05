from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Mapping

from fedbench.core.eval.evalcontext import EvalContext


class Category(StrEnum):
    FIDELITY = "fidelity"
    UTILITY = "utility"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    SCALABILITY = "scalability"


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> Mapping[str, float]: ...
