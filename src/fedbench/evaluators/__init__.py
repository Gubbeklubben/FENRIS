from typing import Mapping

from fedbench.core.eval import Category, Evaluator
from fedbench.core.factory_registry import FactoryRegistry


def register_builtin_evaluators(
    registries: Mapping[str, FactoryRegistry[Evaluator]],
) -> None:

    # Fidelity Evaluators
    registries[Category.FIDELITY].add_builtin(
        "moment_reduction_metrics",
        f"{__package__}.fidelity:MomentReductionMetricsEvaluator",
    )
    registries[Category.FIDELITY].add_builtin(
        "distribution_similarity_metrics",
        f"{__package__}.fidelity:DistributionSimilarityMetricsEvaluator",
    )
    registries[Category.FIDELITY].add_builtin(
        "categorical_tv_mean", f"{__package__}.fidelity:CategoricalTvMeanEvaluator"
    )
    registries[Category.FIDELITY].add_builtin(
        "corr_fro_diff", f"{__package__}.fidelity:CorrFroDiffEvaluator"
    )

    # Utility Evaluators
    registries[Category.UTILITY].add_builtin(
        "tstr", f"{__package__}.utility:TSTREvaluator"
    )

    # Privacy Evaluators
    registries[Category.PRIVACY].add_builtin(
        "direct_overlap_diagnostic",
        f"{__package__}.privacy:DirectOverlapDiagnosticEvaluator",
    )
    registries[Category.PRIVACY].add_builtin(
        "mia_nearest_neighbor_attack",
        f"{__package__}.privacy:MIANearestNeighborAttackEvaluator",
    )
    registries[Category.PRIVACY].add_builtin(
        "aia_supervised_attack", f"{__package__}.privacy:AIASupervisedAttackEvaluator"
    )

    # Fairness Evaluators
    registries[Category.FAIRNESS].add_builtin(
        "fairness", f"{__package__}.fairness:FairnessEvaluator"
    )
