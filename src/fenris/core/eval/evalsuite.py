from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from fenris.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext
from fenris.core.eval.evaluator import Evaluator, MetricSpec


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[Evaluator]):
        self._evaluators = tuple(evaluators)

    def __iter__(self) -> Iterator[Evaluator]:
        yield from self._evaluators

    def get_evaluator_for_metric_key(
        self,
        metric_key: str,
        target_column: str | None,
        sensitive_columns: tuple[str, ...] | None,
    ) -> tuple[Evaluator, MetricSpec]:

        for ev in self._evaluators:
            for key, metric in ev.get_metric_spec_dict(
                target_column, sensitive_columns
            ).items():
                if f"{ev.EVALUATOR_SPEC.category}.{key}" == metric_key:
                    return ev, metric
        raise KeyError(
            f"Specified metric key {metric_key} is not emitted "
            f"by any evaluator in the current evaluation suite."
        )

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
                f"Incorrect output shape for evaluator {evaluator.name}."
                f"\nDeclared: {declared_keys}\nActual: {actual_keys}"
            )
        return {
            f"{evaluator.EVALUATOR_SPEC.category}.{key}": value
            for key, value in raw_metrics.items()
        }

    # noinspection PyMethodMayBeStatic
    def _global_evaluate(
        self, ctx: GlobalEvalContext, evaluators: Iterable[Evaluator]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for ev in evaluators:
            metrics.update(
                self._prefix_and_verify_key_names(
                    ev,
                    ev.global_evaluate(ctx),
                    ctx.target_column,
                    ctx.sensitive_columns,
                )
            )
        return metrics

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        return self._global_evaluate(ctx, self._evaluators)

    def global_evaluate_single(self, ctx: GlobalEvalContext, key: str) -> float:
        """Run only the evaluator that owns `key`, return the single metric value."""
        evaluator, _ = self.get_evaluator_for_metric_key(
            key, ctx.target_column, ctx.sensitive_columns
        )
        metrics = self._global_evaluate(ctx, [evaluator])
        return metrics[key]

    def local_evaluate(self, ctx: LocalEvalContext) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for ev in self._evaluators:
            metrics[ev.name] = ev.local_evaluate(ctx)
        return metrics

    def aggregate(
        self,
        per_client_metrics: Iterable[Mapping[str, Any]],
        target_column: str | None,
        sensitive_columns: tuple[str, ...] | None,
    ) -> dict[str, float]:
        aggregated_metrics: dict[str, float] = {}
        for ev in self._evaluators:
            stats = [client_metrics[ev.name] for client_metrics in per_client_metrics]
            aggregated_metrics.update(
                self._prefix_and_verify_key_names(
                    ev, ev.aggregate(stats), target_column, sensitive_columns
                )
            )
        return aggregated_metrics
