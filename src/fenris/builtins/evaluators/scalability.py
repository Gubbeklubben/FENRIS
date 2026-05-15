"""Scalability evaluators.

Most scalability metrics are *measurements*, not computations: wall-clock time,
byte counts, and round count are accumulated by ScalabilityCollector via the
EventBus and injected directly into aggregated_metrics — they do not pass
through this evaluator at all.

This evaluator handles only ``local_train_seconds_mean``, which is self-reported
by each client via LocalEvalContext and aggregated here as a row-count-weighted
mean.  global_evaluate returns NaN for all keys — scalability has no centralized
analogue.

Registered metric keys
----------------------
  scalability.wall_clock_seconds        — set by ScalabilityCollector
  scalability.bytes_sent                — set by ScalabilityCollector
  scalability.bytes_received            — set by ScalabilityCollector
  scalability.rounds_to_converge        — set by ScalabilityCollector
  scalability.local_train_seconds_mean  — set by this evaluator
"""

from __future__ import annotations

from typing import ClassVar, Iterable

from fenris.builtins.evaluators._helpers import weighted_mean
from fenris.core.eval import Category
from fenris.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext
from fenris.core.eval.evaluator import (
    EvaluationMode,
    Evaluator,
    EvaluatorSpec,
    MetricSpec,
)


class ScalabilityEvaluator(Evaluator):
    """Evaluator for metric family 5: Scalability.

    Handles only the ``local_train_seconds_mean`` aggregation path.
    All other scalability keys are accumulated by ScalabilityCollector and
    merged into aggregated_metrics outside the evaluator pipeline.
    """

    EVALUATOR_SPEC: ClassVar[EvaluatorSpec] = EvaluatorSpec(
        category=Category.SCALABILITY,
        eval_mode=EvaluationMode.FEDERATED,
        metrics=[
            MetricSpec("wall_clock_seconds", default_stop_mode=None),
            MetricSpec("bytes_sent", default_stop_mode=None),
            MetricSpec("bytes_received", default_stop_mode=None),
            MetricSpec("rounds_to_converge", default_stop_mode=None),
            MetricSpec("local_train_seconds_mean", default_stop_mode=None),
        ],
    )

    # ------------------------------------------------------------------
    # Centralized / global path
    # ------------------------------------------------------------------

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        """Return NaN for all keys — scalability has no centralized analogue."""
        return self._nan_result()

    # ------------------------------------------------------------------
    # Federated client-side path
    # ------------------------------------------------------------------

    def local_evaluate(self, ctx: LocalEvalContext) -> tuple[float, int]:
        """Return ``(local_train_seconds, n_train_rows)``
        for weighted-mean aggregation.
        """
        return ctx.local_train_seconds, len(ctx.train_df)

    # ------------------------------------------------------------------
    # Federated server-side path
    # ------------------------------------------------------------------

    def aggregate(self, stats: Iterable[tuple[float, int]]) -> dict[str, float]:
        """Compute row-count-weighted mean of per-client ``local_train_seconds``.

        Clients whose value is NaN are excluded from the mean.
        The other ScalabilityCollector keys are emitted as NaN here.
        Their proper values are injected during aggregate_federated_metrics.
        """
        return {
            **self._nan_result(),
            "local_train_seconds_mean": weighted_mean(stats),
        }
