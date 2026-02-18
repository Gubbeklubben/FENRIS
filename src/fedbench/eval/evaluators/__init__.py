from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from fedbench.eval.evaluators.base import Evaluator
from fedbench.registry import Registry


class Category(StrEnum):
    FIDELITY    = "fidelity"
    UTILITY     = "utility"
    PRIVACY     = "privacy"
    FAIRNESS    = "fairness"
    SCALABILITY = "scalability"


def _validator(evaluator_cls: type[Evaluator]) -> type[Evaluator]:
    if not issubclass(evaluator_cls, Evaluator):
        raise TypeError(f"{evaluator_cls} must be a subclass of {Evaluator}")
    return evaluator_cls


_registries: dict[str, Registry[type[Evaluator]]] = {
    category: Registry(
        group=f"{__package__}.{category}",
        validator=_validator,
    ) for category in Category
}

_registries[Category.FIDELITY].add_builtin(
    "mean_abs_diff",
    f"{__package__}.fidelity:MeanAbsDiffEvaluator"
)
_registries[Category.FIDELITY].add_builtin(
    "std_abs_diff",
    f"{__package__}.fidelity:StdAbsDiffEvaluator"
)
_registries[Category.FIDELITY].add_builtin(
    "categorical_tv_mean",
    f"{__package__}.fidelity:CategoricalTvMeanEvaluator"
)
_registries[Category.FIDELITY].add_builtin(
    "corr_fro_diff",
    f"{__package__}.fidelity:CorrFroDiffEvaluator"
)
_registries[Category.FIDELITY].add_builtin(
    "ks_mean",
    f"{__package__}.fidelity:KsMeanEvaluator"
)
_registries[Category.FIDELITY].add_builtin(
    "wasserstein_mean",
    f"{__package__}.fidelity:WassersteinMeanEvaluator"
)
_registries[Category.FIDELITY].add_builtin(
    "t_stat_mean_abs",
    f"{__package__}.fidelity:TStatMeanAbsEvaluator"
)

registries: Mapping[str, Registry[type[Evaluator]]] = MappingProxyType(_registries)

__all__ = [
    "Evaluator",
    "Category",
    "registries"
]