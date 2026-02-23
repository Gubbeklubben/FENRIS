from collections.abc import Iterable, Mapping
from typing import Self

from fedbench.core.eval.evalcontext import EvalContext
from fedbench.core.eval.evaluator import Category, Evaluator
from fedbench.core.registry import FactoryRegistry


def _get_by_categories(
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        categories: Iterable[str]) -> Iterable[tuple[str, Evaluator]]:

    for category in categories:
        registry = registries[category]
        for name in registry:
            yield f"{category}.{name}", registry.call(name)


def _get_by_names(
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        names: Iterable[str]) -> Iterable[tuple[str, Evaluator]]:

    names = set(names)
    for category in Category:
        registry = registries[category]

        for name in registry:
            if name not in names: continue
            yield f"{category}.{name}", registry.call(name)


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[tuple[str, Evaluator]]):
        self._evaluators = tuple(evaluators)

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, ev in self._evaluators:
            metrics[name] = ev.evaluate(ctx)
        return metrics

    @classmethod
    def default(cls, registries: Mapping[str, FactoryRegistry[Evaluator]]) -> Self:
        return cls.with_evaluator_categories(
            registries, [category.value for category in Category]
        )

    @classmethod
    def with_evaluator_categories(
            cls,
            registries: Mapping[str, FactoryRegistry[Evaluator]],
            categories: Iterable[str]) -> Self:

        return cls(_get_by_categories(registries, categories))

    @classmethod
    def with_evaluator_names(
            cls,
            registries: Mapping[str, FactoryRegistry[Evaluator]],
            names: Iterable[str]) -> Self:

        return cls(_get_by_names(registries, names))