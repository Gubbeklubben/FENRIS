"""Defines the Synthesizer ABC.

Classes
-------
Synthesizer
    Abstract base class for synthesizer implementations.
GlobalInitArtifacts
    Simple container for preprocessing artifacts.
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pandas import DataFrame

from fedbench.core.algorithm.context import (
    GlobalInitContext,
    SampleContext,
    TrainContext,
)
from fedbench.core.component import Component
from fedbench.core.payload import ArraysTarget, Payload


@dataclass(frozen=True)
class GlobalInitArtifacts:
    """A simple container for preprocessing output.

    Attributes
    ----------
    coordinator : Payload | None
        The content of this field, if any, is injected into the coordinator's
        attach_global_init_artifacts.
    synthesizer : Payload | None
        The content of this field, if any, is stored by the framework,
        and always attached to the context passed to a Synthesizer's
        train and sample methods.
    """

    coordinator: Payload | None = None
    synthesizer: Payload | None = None


class Synthesizer(Component):
    """The framework view of the model to train and sample from.

    Attributes
    ----------
    SUPPORTED_COORDINATORS : set[str]
        Names of supported coordinators. Looked up at class level. If not overridden,
        it is an empty set.
    """

    SUPPORTED_COORDINATORS: ClassVar[set[str]] = set()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "sample" not in cls.__dict__:
            return
        original = cls.__dict__["sample"]

        @functools.wraps(original)
        def wrapper(
            self: Synthesizer, request: Payload, context: SampleContext
        ) -> DataFrame:
            synthetic_df = original(self, request, context)
            if not isinstance(synthetic_df, DataFrame):
                raise TypeError(
                    f"{str(self)}.sample() must return a DataFrame "
                    f"(got {type(synthetic_df)})."
                )
            if synthetic_df.empty:
                raise ValueError(
                    f"DataFrame returned from {str(self)}.sample() is empty. "
                    f"Expected {context.num_rows} synthetic rows."
                )
            if len(synthetic_df) != context.num_rows:
                raise ValueError(
                    f"DataFrame returned from {str(self)}.sample() has "
                    f"an incorrect number of synthetic rows. "
                    f"Expected: {context.num_rows}. Actual: {len(synthetic_df)}."
                )
            schema_columns = {col.name for col in context.schema.columns}
            if schema_columns != set(synthetic_df.columns):
                raise ValueError(
                    f"DataFrame returned from {str(self)}.sample() "
                    f"does not match schema."
                    f"\nSchema columns: {sorted(schema_columns)}"
                    f"\nDataFrame columns: {sorted(synthetic_df.columns)}"
                )
            return synthetic_df

        setattr(cls, "sample", wrapper)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def arrays_target(self) -> ArraysTarget:
        """Array deserialization target.

        Decides the runtime type of the arrays field of
        `fedbench.core.payload.Payload` instances.

        Returns
        -------
        `fedbench.core.payload.ArraysTarget`
            numpy | torch
        """

    @abstractmethod
    def global_init(
        self,
        df: DataFrame,
        context: GlobalInitContext,
    ) -> GlobalInitArtifacts:
        """Preprocessing hook.

        Called before initiating a federated simulation.

        Parameters
        ----------
        df : `pandas.DataFrame`
            The union of all train partitions.
        context : `fedbench.core.algorithm.context.GlobalInitContext`
            A context object holding relevant information like the table schema and the
            derived seed to use for stochastic operations during initialization.

        Returns
        -------
        `fedbench.core.algorithm.context.GlobalInitArtifacts`
            The resulting preprocessing artifacts.
        """

    @abstractmethod
    def train(
        self,
        request: Payload,
        df: DataFrame,
        context: TrainContext,
    ) -> Payload:
        """Respond to a coordinator train request.

        Parameters
        ----------
        request : `fedbench.core.payload.Payload`
            Incoming request from the active coordinator.
        df : `pandas.DataFrame`
            The local train partition.
        context : `fedbench.core.algorithm.context.TrainContext`
            A context object holding relevant information like preprocessing artifacts.
            Includes a client local read/write storage.

        Returns
        -------
        `fedbench.core.payload.Payload`
            Response content.
        """

    @abstractmethod
    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> DataFrame:
        """Sample synthetic data.

        Parameters
        ----------
        request : `fedbench.core.payload.Payload`
            Most recent global train artifacts published by the active coordinator.
        context : `fedbench.core.algorithm.context.SampleContext`
            A context object holding relevant information like preprocessing artifacts.
            Includes a client local read/write storage.

        Returns
        -------
        `pandas.DataFrame`
            The sampled synthetic data.
        """
