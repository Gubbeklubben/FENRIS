from collections.abc import Iterable, Mapping
from typing import Self

from fedbench.core.eval.evalcontext import EvalContext
from fedbench.core.eval.evaluator import Category, Evaluator
from fedbench.core.factory_registry import FactoryRegistry


def _get_evaluators(
    registries: Mapping[str, FactoryRegistry[Evaluator]],
    categories: Iterable[str] = tuple(Category),
    names: Iterable[str] = (),
) -> Iterable[tuple[str, str, Evaluator]]:

    names = set(names)
    for category in categories:
        registry = registries[category]
        for name in registry:
            if names and name not in names:
                continue
            yield name, category, registry.call(name)


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[tuple[str, str, Evaluator]]):
        self._evaluators = tuple(evaluators)

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for _, category, ev in self._evaluators:
            for metric, value in ev.evaluate(ctx).items():
                metrics[f"{category}.{metric}"] = value
        return metrics

    @classmethod
    def default(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
    ) -> Self:
        return cls.with_evaluator_categories(registries, tuple(Category))

    @classmethod
    def with_evaluator_categories(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        categories: Iterable[str],
    ) -> Self:
        return cls(_get_evaluators(registries, categories=categories))

    @classmethod
    def with_evaluator_names(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        names: Iterable[str],
    ) -> Self:
        return cls(_get_evaluators(registries, names=names))
