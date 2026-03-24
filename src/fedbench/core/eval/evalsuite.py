from collections.abc import Iterable, Mapping
from typing import Any, Self

from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext
from fedbench.core.eval.evaluator import Category, Evaluator
from fedbench.runtime.registry import FactoryRegistry


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[Evaluator]):
        self._evaluators = tuple(evaluators)

    @classmethod
    def default(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
    ) -> Self:
        return cls(cls._get_evaluators(registries))

    @classmethod
    def with_evaluator_categories(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        categories: Iterable[Category],
    ) -> Self:
        return cls(cls._get_evaluators(registries, categories=categories))

    @classmethod
    def with_evaluator_names(
        cls,
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        names: Iterable[str],
    ) -> Self:
        return cls(cls._get_evaluators(registries, names=names))

    @staticmethod
    def _get_evaluators(
        registries: Mapping[str, FactoryRegistry[Evaluator]],
        categories: Iterable[Category] = tuple(Category),
        names: Iterable[str] = (),
    ) -> Iterable[Evaluator]:
        names = set(names)
        for category in categories:
            registry = registries[category]
            for name in registry:
                if names and name not in names:
                    continue
                yield registry.call(name)

    # noinspection PyMethodMayBeStatic
    def _prefix_and_verify_key_names(
        self,
        evaluator: Evaluator,
        raw_metrics: dict[str, float],
        target_column: str | None,
        sensitive_columns: tuple[str, ...] | None,
    ) -> dict[str, float]:
        declared_keys = set(evaluator.get_metric_keys(target_column, sensitive_columns))
        actual_keys = set(raw_metrics.keys())
        if actual_keys != declared_keys:
            raise ValueError(
                f"Incorrect output shape for evaluator {evaluator.metadata.name}."
                f"\nDeclared: {declared_keys}\nActual: {actual_keys}"
            )
        return {
            f"{evaluator.metadata.category}.{key}": value  # nofmt
            for key, value in raw_metrics.items()
        }

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for ev in self._evaluators:
            metrics.update(
                self._prefix_and_verify_key_names(
                    ev,
                    ev.global_evaluate(ctx),
                    ctx.target_column,
                    ctx.sensitive_columns,
                )
            )
        return metrics

    def local_evaluate(self, ctx: LocalEvalContext) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for ev in self._evaluators:
            metrics[ev.metadata.name] = ev.local_evaluate(ctx)
        return metrics

    def aggregate(
        self,
        per_client_metrics: Iterable[Mapping[str, Any]],
        target_column: str | None,
        sensitive_columns: tuple[str, ...] | None,
    ) -> dict[str, float]:
        aggregated_metrics: dict[str, float] = {}
        for ev in self._evaluators:
            stats = [
                client_metrics[ev.metadata.name]
                for client_metrics in per_client_metrics
            ]
            aggregated_metrics.update(
                self._prefix_and_verify_key_names(
                    ev, ev.aggregate(stats), target_column, sensitive_columns
                )
            )
        return aggregated_metrics
