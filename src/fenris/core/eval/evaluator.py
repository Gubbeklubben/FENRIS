import math
import re
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Flag, StrEnum, auto
from typing import Any, ClassVar, Literal

from fenris.core.component import Component
from fenris.core.eval.evalcontext import GlobalEvalContext, LocalEvalContext


def normalize_key(text: str) -> str:
    """Convert *text* to a lowercase, underscore-separated metric key.

    Non-alphanumeric character runs are replaced with a single underscore.

    Parameters
    ----------
    text : str
        Raw metric name to normalize

    Returns
    -------
    str
    """
    return re.sub(r"[^a-z_]+", "_", text.lower())


class Category(StrEnum):
    """Evaluation category labels used to prefix fully qualified metric keys."""

    FIDELITY = "fidelity"
    UTILITY = "utility"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    SCALABILITY = "scalability"


class EvaluationMode(Flag):
    """Flag indicating whether an evaluator supports centralized or federated mode."""

    CENTRALIZED = auto()
    FEDERATED = auto()
    # noinspection PyTypeChecker
    BOTH = CENTRALIZED | FEDERATED


@dataclass(frozen=True)
class MetricSpec:
    """Metadata for a single metric emitted by an `Evaluator`.

    Attributes
    ----------
    key : str
        Metric key used as the suffix after the category prefix.
    default_stop_mode : {"min", "max"} or None
        Default optimization direction for early stopping; ``None`` if no sensible
        default exists
    suffix_type : {"sensitive", "target", None}
        Whether the metric is expanded once per sensitive column
        (``"sensitive"``), once for the target column (``"target"``), or
        emitted as-is (``None``).
    """

    key: str
    default_stop_mode: Literal["min", "max"] | None = "min"
    suffix_type: Literal["sensitive", "target", None] = None


@dataclass(frozen=True)
class EvaluatorSpec:
    """Metadata declaring an evaluator's category, mode, and metrics.

    Attributes
    ----------
    category : Category
        Evaluation category this evaluator belongs to.
    eval_mode : EvaluationMode
        Whether the evaluator supports centralized, federated, or both modes.
    metrics : list[MetricSpec]
        Metadata for each metric the evaluator emits.
    """

    category: Category
    eval_mode: EvaluationMode
    metrics: list[MetricSpec]


class Evaluator(Component):
    """Abstract base class for evaluators.

    Evaluators compute quality metrics for synthetic data, supporting both
    centralized evaluation (via `global_evaluate`) and federated evaluation
    (via `local_evaluate` on clients followed by `aggregate` on the server).
    """

    # [scaffold] required_cls_var
    EVALUATOR_SPEC: ClassVar[EvaluatorSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce that every Evaluator subclass declares ``EVALUATOR_SPEC``.

        Raises
        ------
        TypeError
            If ``EVALUATOR_SPEC`` is not declared on the subclass.
        """
        super().__init_subclass__(**kwargs)
        if "EVALUATOR_SPEC" not in cls.__dict__:
            raise TypeError(
                f"{cls}: Evaluator subclass must declare class variable EVALUATOR_SPEC."
            )

    def get_metric_spec_dict(
        self,
        target_column: str | None = None,
        sensitive_columns: tuple[str, ...] | None = None,
    ) -> dict[str, MetricSpec]:
        """Build a mapping from fully qualified metric key to `MetricSpec`.

        Sensitive- and target-suffixed metrics are expanded from their template
        specs using the provided column names.

        Parameters
        ----------
        target_column : str or None, optional
            Target column name.
        sensitive_columns : tuple[str, ...] or None, optional
            Sensitive column names.

        Returns
        -------
        dict[str, MetricSpec]
        """
        specs: dict[str, MetricSpec] = {}
        for metric in self.EVALUATOR_SPEC.metrics:
            if sensitive_columns and metric.suffix_type == "sensitive":
                for suffix in sensitive_columns:
                    specs[f"{metric.key}.{normalize_key(suffix)}"] = metric
            elif target_column and metric.suffix_type == "target":
                specs[f"{metric.key}.{normalize_key(target_column)}"] = metric
            else:
                specs[metric.key] = metric
        return specs

    def get_metric_keys(
        self,
        target_column: str | None = None,
        sensitive_columns: tuple[str, ...] | None = None,
    ) -> Iterable[str]:
        """Return all fully qualified metric keys for this evaluator.

        Parameters
        ----------
        target_column : str or None, optional
            Target column name for expanding target-suffixed metrics.
        sensitive_columns : tuple[str, ...] or None, optional
            Sensitive column names for expanding sensitive-suffixed metrics.

        Returns
        -------
        Iterable[str]
        """
        return self.get_metric_spec_dict(target_column, sensitive_columns).keys()

    def _nan_result(self) -> dict[str, float]:
        """Return a dict of NaN values keyed by this evaluator's metric keys.

        Returns
        -------
        dict[str, float]
            Every metric key mapped to ``float("nan")``.
        """
        return {key: math.nan for key in self.get_metric_keys()}

    @abstractmethod
    def global_evaluate(self, ctx: GlobalEvalContext) -> dict[str, float]:
        """Centralized evaluation using the full synthetic and holdout sets.

        Parameters
        ----------
        ctx : GlobalEvalContext
            Evaluation context containing the synthetic data and holdout set.

        Returns
        -------
        dict[str, float]
            Metric key to value mapping. Keys must match those declared in
            `metadata`.
        """

    @abstractmethod
    def local_evaluate(self, ctx: LocalEvalContext) -> Any:
        """Federated evaluation on one client's local data.

        Called on each client during federated evaluation. The return value
        is opaque to the framework and passed verbatim to `aggregate`.

        Parameters
        ----------
        ctx : LocalEvalContext
            Evaluation context containing the client's local partitions.

        Returns
        -------
        Any
            Intermediate statistics to be aggregated server-side.
        """

    @abstractmethod
    def aggregate(self, stats: Iterable[Any]) -> dict[str, float]:
        """Aggregate per-client local evaluation results on the server.

        Parameters
        ----------
        stats : Iterable[Any]
            Collection of values returned by `local_evaluate` across all clients.

        Returns
        -------
        dict[str, float]
            Aggregated metric key to value mapping. Keys must match those
            declared in `metadata`.
        """
