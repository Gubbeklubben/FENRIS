from typing import Mapping

from fedbench.core.eval import Category, Evaluator
from fedbench.core.factory_registry import FactoryRegistry


def register_builtin_evaluators(
        registries: Mapping[str, FactoryRegistry[Evaluator]]) -> None:

    registries[Category.FIDELITY].add_builtin(
        "moment_reduction_metrics",
        f"{__package__}.fidelity:MomentReductionMetricsEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "distribution_similarity_metrics",
        f"{__package__}.fidelity:DistributionSimilarityMetricsEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "categorical_tv_mean",
        f"{__package__}.fidelity:CategoricalTvMeanEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "corr_fro_diff",
        f"{__package__}.fidelity:CorrFroDiffEvaluator"
    )