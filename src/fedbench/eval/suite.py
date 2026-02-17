from typing import Iterable, Dict

from fedbench.eval.context import EvalContext
from fedbench.eval.evaluators import (
    Evaluator,
    Category,
    registries as evaluator_regs
)


def _get_by_categories(categories: Iterable[str]) -> Iterable[Evaluator]:
    for category in categories:
        registry = evaluator_regs[category]
        for name, _ in registry:
            factory = registry.load(name)
            yield factory()


def _get_by_names(names: Iterable[str]) -> Iterable[Evaluator]:
    names = set(names)
    for category in Category:
        registry = evaluator_regs[category]

        for name, _ in registry:
            if name not in names: continue
            factory = registry.load(name)
            yield factory()


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[Evaluator]):
        self._evaluators = tuple(evaluators)

    def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for ev in self._evaluators:
            metrics[f"{ev.category}.{ev.name}"] = ev.evaluate(ctx)
        return metrics

    @classmethod
    def default(cls):
        return cls.with_evaluator_names(tuple(str(Category)))

    @classmethod
    def with_evaluator_categories(cls, categories: Iterable[str]):
        return cls(_get_by_categories(categories))

    @classmethod
    def with_evaluator_names(cls, names: Iterable[str]):
        return cls(_get_by_names(names))