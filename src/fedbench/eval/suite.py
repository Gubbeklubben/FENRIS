from typing import Iterable, Dict

from fedbench.eval.context import EvalContext
from fedbench.eval.evaluators import (
    Evaluator,
    Category,
    registries as evaluator_regs
)


def _get_by_categories(categories: Iterable[str]) -> Iterable[tuple[str, Evaluator]]:
    for category in categories:
        registry = evaluator_regs[category]
        for metadata in registry:
            factory = registry.load(metadata.name)
            yield f"{category}.{metadata.name}", factory()


def _get_by_names(names: Iterable[str]) -> Iterable[tuple[str, Evaluator]]:
    names = set(names)
    for category in Category:
        registry = evaluator_regs[category]

        for metadata in registry:
            if metadata.name not in names: continue
            factory = registry.load(metadata.name)
            yield f"{category}.{metadata.name}", factory()


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[tuple[str, Evaluator]]):
        self._evaluators = tuple(evaluators)

    def evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, ev in self._evaluators:
            metrics[name] = ev.evaluate(ctx)
        return metrics

    @classmethod
    def default(cls):
        return cls.with_evaluator_categories([category.value for category in Category])

    @classmethod
    def with_evaluator_categories(cls, categories: Iterable[str]):
        return cls(_get_by_categories(categories))

    @classmethod
    def with_evaluator_names(cls, names: Iterable[str]):
        return cls(_get_by_names(names))