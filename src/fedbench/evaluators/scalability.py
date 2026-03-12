"""
fedbench/evaluators/scalability.py

ScalabilityEvaluator — metric family 5 (spec §9).

Design contract
---------------
Scalability metrics are *measurements*, not computations.  The orchestration
layer (ScalabilityCollector via EventBus) accumulates timing and byte counts
during the federation run.  The pipeline reads those measurements in
aggregate_federated_metrics and merges them directly into aggregated_metrics
via ScalabilityCollector.get_metrics().

This evaluator's role is therefore minimal:

  - global_evaluate  always returns NaN for all keys; scalability has no
                     centralized analogue (spec §15.5, table 5).
  - local_evaluate   returns each client's self-reported local_train_seconds
                     and row count for server-side weighted averaging.
  - aggregate        computes local_train_seconds_mean from per-client payloads.
                     The four canonical server-side keys are not available here
                     and are not emitted; they arrive via get_metrics().

Registered metric keys (spec §9 / metric registry)
---------------------------------------------------
  scalability.wall_clock_seconds        total federation wall-clock time (§9.1)
  scalability.bytes_sent                server→client bytes, accumulated (§9.2)
  scalability.bytes_received            client→server bytes, accumulated (§9.2)
  scalability.rounds_to_converge        early-stop round, or num_rounds (§9.3)
  scalability.local_train_seconds_mean  weighted mean client training time

Data flow
---------
The five keys reach metrics.federated.json via two separate paths that are
merged in aggregate_federated_metrics:

  ScalabilityCollector.get_metrics()
      wall_clock_seconds, bytes_sent, bytes_received, rounds_to_converge
      ──────────────────────────────────────────────────────────────────
      Measured server-side by ScalabilityCollector observing RoundStarted,
      RoundCompleted, ServerRequest and ClientReply events.  Injected
      directly into aggregated_metrics without passing through any evaluator.

  EvaluationSuite.aggregate() → ScalabilityEvaluator.aggregate()
      local_train_seconds_mean
      ──────────────────────────────────────────────────────────────────
      Each client self-reports local_train_seconds via LocalEvalContext
      (measured in FlwrClient.train(), stored in FlwrClient._timing, read in
      FlwrClient.evaluate()).  The server computes a row-count-weighted mean.

Scalability metrics are inherently federated and have no centralized analogue.
global_evaluate returns NaN for all keys so that metrics.centralized.json
preserves a stable shape.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from fedbench.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext
from fedbench.core.eval.evaluator import Evaluator


class ScalabilityEvaluator(Evaluator):
    """
    Evaluator for metric family 5: Scalability.

    Zero-compute: measurements are collected by ScalabilityCollector and
    injected into aggregated_metrics by the pipeline.  This class handles
    only the local_train_seconds aggregation path, which does pass through
    the standard evaluator interface.  See module docstring for full data
    flow.
    """

    # ------------------------------------------------------------------
    # Centralized / global path
    # ------------------------------------------------------------------

    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        """
        Scalability metrics have no centralized analogue (spec §15.5, table 5).

        Returns NaN for all keys so that metrics.centralized.json maintains
        a stable shape.  The real values are written to metrics.federated.json
        via ScalabilityCollector.get_metrics() in aggregate_federated_metrics.
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

    def local_evaluate(self, ctx: LocalEvalContext) -> dict[str, Any]:
        """
        Return the client's self-reported training time and row count.

        local_train_seconds is measured in FlwrClient.train() and stored
        on FlwrClient._timing, from where FlwrClient.evaluate() reads it
        into LocalEvalContext.  n_train_rows is used as the aggregation
        weight in aggregate().

        Return shape
        ------------
        {
            "local_train_seconds": float,   # math.nan if train() not yet called
            "n_train_rows":        float,   # always >= 0
        }
        """
        return {
            "local_train_seconds": ctx.local_train_seconds,
            "n_train_rows": float(len(ctx.train_df)),
        }

    # ------------------------------------------------------------------
    # Federated server-side path
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(stats: Iterable[dict[str, Any]]) -> dict[str, float]:
        """
        Compute a row-count-weighted mean of per-client local_train_seconds.

        Receives the dicts emitted by local_evaluate from each participating
        client.  Clients whose local_train_seconds is NaN (train() was not
        called before evaluate()) are excluded from the weighted mean.

        The four server-side canonical keys (wall_clock_seconds, bytes_sent,
        bytes_received, rounds_to_converge) are not available here — they are
        accumulated by ScalabilityCollector and merged into aggregated_metrics
        separately.  This method does not emit them.

        Return shape
        ------------
        {
            "local_train_seconds_mean": float,  # math.nan if no valid clients
        }
        """

        total_weight = 0.0
        weighted_sum = 0.0

        for entry in stats:
            if not isinstance(entry, dict):
                continue
            secs = entry.get("local_train_seconds", math.nan)
            weight = entry.get("n_train_rows", 0.0)
            if not math.isnan(secs) and weight > 0:
                weighted_sum += float(secs) * float(weight)
                total_weight += float(weight)

        local_train_seconds_mean = (
            weighted_sum / total_weight if total_weight > 0 else math.nan
        )

        return {
            "local_train_seconds_mean": local_train_seconds_mean,
        }
