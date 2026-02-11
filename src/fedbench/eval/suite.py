# fed_synth_bench/eval/suite.py
from typing import Iterable, Dict

from .evaluators.base import Evaluator
from .context import EvalContext

from .evaluators.fidelity import BasicFidelityEvaluator
from .evaluators.utility import TSTREvaluator
from .evaluators.privacy import PrivacyEvaluator
from .evaluators.fairness import FairnessEvaluator
from .evaluators.scalability import ScalabilityEvaluator
from .evaluators.fidelity_extended import ExtendedFidelityEvaluator


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[Evaluator]):
        self.evaluators = list(evaluators)

    def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for ev in self.evaluators:
            metrics.update(ev.evaluate(ctx))
        return metrics

    @staticmethod
    def default(selected: Iterable[str]):
        mapping = {
            "fidelity": BasicFidelityEvaluator(),
            "fidelity_ext": ExtendedFidelityEvaluator(),
            "utility": TSTREvaluator(),
            "privacy": PrivacyEvaluator(),
            "fairness": FairnessEvaluator(),
            "scalability": ScalabilityEvaluator(),
        }
        return EvaluationSuite(mapping[k] for k in selected)
