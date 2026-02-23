from typing import Mapping

from fedbench.core.eval import Category, Evaluator
from fedbench.core.registry import FactoryRegistry


def register_builtin_evaluators(
        registries: Mapping[str, FactoryRegistry[Evaluator]]) -> None:

    registries[Category.FIDELITY].add_builtin(
        "mean_abs_diff",
        f"{__package__}.fidelity:MeanAbsDiffEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "std_abs_diff",
        f"{__package__}.fidelity:StdAbsDiffEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "categorical_tv_mean",
        f"{__package__}.fidelity:CategoricalTvMeanEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "corr_fro_diff",
        f"{__package__}.fidelity:CorrFroDiffEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "ks_mean",
        f"{__package__}.fidelity:KsMeanEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "wasserstein_mean",
        f"{__package__}.fidelity:WassersteinMeanEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "t_stat_mean_abs",
        f"{__package__}.fidelity:TStatMeanAbsEvaluator"
    )