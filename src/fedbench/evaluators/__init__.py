from fedbench.core.eval import Evaluator
from fedbench.runtime.registry import FactoryRegistry


def register_builtin_evaluators(
    registry: FactoryRegistry[Evaluator],
) -> None:

    # Fidelity Evaluators
    registry.add_builtin(
        "moment_reduction_metrics",
        f"{__package__}.fidelity:MomentReductionMetricsEvaluator",
    )
    registry.add_builtin(
        "distribution_similarity_metrics",
        f"{__package__}.fidelity:DistributionSimilarityMetricsEvaluator",
    )
    registry.add_builtin(
        "categorical_tv_mean",
        f"{__package__}.fidelity:CategoricalTvMeanEvaluator",
    )
    registry.add_builtin(
        "corr_fro_diff",
        f"{__package__}.fidelity:CorrFroDiffEvaluator",
    )

    # Utility Evaluators
    registry.add_builtin(
        "tstr",
        f"{__package__}.utility:TSTREvaluator",
    )

    # Privacy Evaluators
    registry.add_builtin(
        "direct_overlap_diagnostic",
        f"{__package__}.privacy:DirectOverlapDiagnosticEvaluator",
    )
    registry.add_builtin(
        "mia_nearest_neighbor_attack",
        f"{__package__}.privacy:MIANearestNeighborAttackEvaluator",
    )
    registry.add_builtin(
        "aia_supervised_attack",
        f"{__package__}.privacy:AIASupervisedAttackEvaluator",
    )

    # Fairness Evaluators
    registry.add_builtin(
        "fairness",
        f"{__package__}.fairness:FairnessEvaluator",
    )

    # Fairness Evaluators
    registry.add_builtin(
        "scalability",
        f"{__package__}.scalability:ScalabilityEvaluator",
    )
