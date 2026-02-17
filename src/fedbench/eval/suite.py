# fed_synth_bench/eval/suite.py
from typing import Iterable, Dict

from fedbench.eval.evaluators.base import Evaluator
from fedbench.eval.context import EvalContext
from fedbench.eval.evaluators.fidelity import MeanAbsDiffEvaluator, StdAbsDiffEvaluator, CorrFroDiffEvaluator, \
    CategoricalTvMeanEvaluator, KsMeanEvaluator, WassersteinMeanEvaluator, TStatMeanAbsEvaluator


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[Evaluator]):
        self.evaluators = list(evaluators)

    def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for ev in self.evaluators:
            metrics[f"{ev.category}.{ev.name}"] = ev.evaluate(ctx)
        return metrics

    @staticmethod
    def all_fidelity_evaluators():
        return EvaluationSuite([
            MeanAbsDiffEvaluator(),
            StdAbsDiffEvaluator(),
            CorrFroDiffEvaluator(),
            CategoricalTvMeanEvaluator(),
            KsMeanEvaluator(),
            WassersteinMeanEvaluator(),
            TStatMeanAbsEvaluator(),
        ])