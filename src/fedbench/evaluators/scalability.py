"""
Scalability evaluators (metric family 5, spec §9).

Most scalability metrics are *measurements*, not computations: wall-clock time,
byte counts, and round count are accumulated by ScalabilityCollector via the
EventBus and injected directly into aggregated_metrics — they do not pass
through this evaluator at all.

This evaluator handles only ``local_train_seconds_mean``, which is self-reported
by each client via LocalEvalContext and aggregated here as a row-count-weighted
mean.  global_evaluate returns NaN for all keys (scalability has no centralized
analogue, spec §15.5 table 5).

Registered metric keys (spec §9)
---------------------------------
  scalability.wall_clock_seconds        — set by ScalabilityCollector
  scalability.bytes_sent                — set by ScalabilityCollector
  scalability.bytes_received            — set by ScalabilityCollector
  scalability.rounds_to_converge        — set by ScalabilityCollector
  scalability.local_train_seconds_mean  — set by this evaluator
"""

from __future__ import annotations

import math
from typing import Iterable

from fedbench.core.eval import Category
from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext
from fedbench.core.eval.evaluator import (
    EvaluationMode,
    Evaluator,
    EvaluatorDescriptor,
    MetricDescriptor,
)
from fedbench.evaluators._helpers import weighted_mean


class ScalabilityEvaluator(Evaluator):
    """Evaluator for metric family 5: Scalability.

    Handles only the ``local_train_seconds_mean`` aggregation path.
    All other scalability keys are accumulated by ScalabilityCollector and
    merged into aggregated_metrics outside the evaluator pipeline.
    """

    @property
    def metadata(self) -> EvaluatorDescriptor:
        return EvaluatorDescriptor(
            name="scalability",
            category=Category.SCALABILITY,
            eval_mode=EvaluationMode.FEDERATED,
            metrics=[
                MetricDescriptor("wall_clock_seconds", default_stop_mode=None),
                MetricDescriptor("bytes_sent", default_stop_mode=None),
                MetricDescriptor("bytes_received", default_stop_mode=None),
                MetricDescriptor("rounds_to_converge", default_stop_mode=None),
                MetricDescriptor("local_train_seconds_mean", default_stop_mode=None),
            ],
        )

    # ------------------------------------------------------------------
    # Centralized / global path
    # ------------------------------------------------------------------

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        """Return NaN for all keys
        — scalability has no centralized analogue (spec §15.5).
        """
        return {
            "wall_clock_seconds": math.nan,
            "bytes_sent": math.nan,
            "bytes_received": math.nan,
            "rounds_to_converge": math.nan,
            "local_train_seconds_mean": math.nan,
        }

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
        The four ScalabilityCollector keys are not emitted here.
        """
        return {
            "local_train_seconds_mean": weighted_mean(stats),
        }
