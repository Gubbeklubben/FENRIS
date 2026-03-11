from collections.abc import Iterable, Mapping
from typing import Any, Self

from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext
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

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for _, category, ev in self._evaluators:
            for key, value in ev.global_evaluate(ctx).items():
                metrics[f"{category}.{key}"] = value
        return metrics

    def local_evaluate(self, ctx: LocalEvalContext) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name, _, ev in self._evaluators:
            metrics[name] = ev.local_evaluate(ctx)
        return metrics

    def aggregate(
        self, per_client_metrics: Iterable[Mapping[str, Any]]
    ) -> dict[str, float]:
        aggregated_metrics: dict[str, float] = {}
        for name, category, ev in self._evaluators:
            stats = [client_metrics[name] for client_metrics in per_client_metrics]
            for key, value in ev.aggregate(stats).items():
                aggregated_metrics[f"{category}.{key}"] = value
        return aggregated_metrics

    @classmethod
    def default(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
    ) -> Self:
        return cls(_get_evaluators(registries))

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
