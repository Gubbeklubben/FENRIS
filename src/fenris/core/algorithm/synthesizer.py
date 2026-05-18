"""Defines the Synthesizer ABC.

Classes
-------
Synthesizer
    Abstract base class for synthesizer implementations.
GlobalInitArtifacts
    Simple container for preprocessing artifacts.
GlobalInitContext
    Context passed to `Synthesizer.global_init`.
TrainContext
    Context passed to `Synthesizer.train`.
SampleContext
    Context passed to `Synthesizer.sample`.
"""

from __future__ import annotations

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pandas import DataFrame

from fenris.core.component import Component
from fenris.core.data import TableSchema
from fenris.core.payload import ArraysTarget, Payload


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


@dataclass(frozen=True)
class _Context:
    """Base context.

    Attributes
    ----------
    coordinator : str
        Name of the active coordinator.
    seed : int
        Seed to use for stochastic operations.
    schema : TableSchema
        Schema classifying the dataset that will be used for the run.
    """

    coordinator: str
    seed: int
    schema: TableSchema


@dataclass(frozen=True)
class GlobalInitContext(_Context):
    """Context passed to `Synthesizer.global_init`."""


@dataclass(frozen=True)
class TrainContext(_Context):
    """Context passed to `Synthesizer.train`.

    Attributes
    ----------
    global_init_artifacts : Payload or None
        Synthesizer-side artifacts produced by `Synthesizer.global_init`, or
        ``None`` if global initialization produced no synthesizer artifacts.
    client_storage : Payload or None
        Persistent per-client key/value store. The synthesizer may read from
        and write to this across training rounds. ``None`` on the first round.
    """

    global_init_artifacts: Payload | None
    client_storage: Payload | None


@dataclass(frozen=True)
class SampleContext(_Context):
    """Context passed to `Synthesizer.sample`.

    Attributes
    ----------
    global_init_artifacts : Payload or None
        Synthesizer-side artifacts produced by `Synthesizer.global_init`, or
        ``None`` if global initialization produced no synthesizer artifacts.
    client_storage : Payload or None
        Persistent per-client store accumulated during training, or ``None``
        if the synthesizer wrote nothing to storage.
    num_rows : int
        Number of synthetic rows to generate.
    """

    global_init_artifacts: Payload | None
    client_storage: Payload | None
    num_rows: int


class Synthesizer(Component):
    """The framework view of the model to train and sample from.

    Attributes
    ----------
    SUPPORTED_COORDINATORS : ClassVar[set[str]]
        Names of supported coordinators, a required class attribute.
    """

    # [scaffold] required_cls_var
    SUPPORTED_COORDINATORS: ClassVar[set[str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap ``sample`` in subclasses to validate the returned DataFrame.

        Raises
        ------
        TypeError
            If ``SUPPORTED_COORDINATORS`` is not declared or does not support
            membership tests.
        """
        super().__init_subclass__(**kwargs)
        if "SUPPORTED_COORDINATORS" not in cls.__dict__:
            raise TypeError(
                f"{cls}: Synthesizer subclass must declare class attribute "
                "SUPPORTED_COORDINATORS."
            )
        if not hasattr(cls.SUPPORTED_COORDINATORS, "__contains__"):
            raise TypeError(
                f"{cls}: The value of SUPPORTED_COORDINATORS must support "
                "membership tests. Using a set is recommended."
            )
        if "sample" not in cls.__dict__:
            return
        original = cls.__dict__["sample"]

        from pandas import DataFrame

        @functools.wraps(original)
        def wrapper(
            self: Synthesizer, request: Payload, context: SampleContext
        ) -> DataFrame:
            synthetic_df = original(self, request, context)
            if not isinstance(synthetic_df, DataFrame):
                raise TypeError(
                    f"{self!s}.sample() must return a DataFrame "
                    f"(got {type(synthetic_df)})."
                )
            if synthetic_df.empty:
                raise ValueError(
                    f"DataFrame returned from {self!s}.sample() is empty. "
                    f"Expected {context.num_rows} synthetic rows."
                )
            if len(synthetic_df) != context.num_rows:
                raise ValueError(
                    f"DataFrame returned from {self!s}.sample() has "
                    f"an incorrect number of synthetic rows. "
                    f"Expected: {context.num_rows}. Actual: {len(synthetic_df)}."
                )
            schema_columns = {col.name for col in context.schema.columns}
            if schema_columns != set(synthetic_df.columns):
                raise ValueError(
                    f"DataFrame returned from {self!s}.sample() "
                    f"does not match schema."
                    f"\nSchema columns: {sorted(schema_columns)}"
                    f"\nDataFrame columns: {sorted(synthetic_df.columns)}"
                )
            return synthetic_df

        setattr(cls, "sample", wrapper)

    def __repr__(self) -> str:
        """Return ``<ClassName>`` string representation.

        Returns
        -------
        str
            ``<ClassName>`` where *ClassName* is the concrete subclass name.
        """
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def arrays_target(self) -> ArraysTarget:
        """Array deserialization target.

        Decides the runtime type of the arrays field of
        `fenris.core.payload.Payload` instances.

        Returns
        -------
        `fenris.core.payload.ArraysTarget`
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
        context : `fenris.core.algorithm.GlobalInitContext`
            A context object holding relevant information like the table schema and the
            derived seed to use for stochastic operations during initialization.

        Returns
        -------
        `fenris.core.algorithm.context.GlobalInitArtifacts`
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
        request : `fenris.core.payload.Payload`
            Incoming request from the active coordinator.
        df : `pandas.DataFrame`
            The local train partition.
        context : `fenris.core.algorithm.TrainContext`
            A context object holding relevant information like preprocessing artifacts.
            Includes a client local read/write storage.

        Returns
        -------
        `fenris.core.payload.Payload`
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
        request : `fenris.core.payload.Payload`
            Most recent global train artifacts published by the active coordinator.
        context : `fenris.core.algorithm.SampleContext`
            A context object holding relevant information like preprocessing artifacts.
            Includes a client local read/write storage.

        Returns
        -------
        `pandas.DataFrame`
            The sampled synthetic data.
        """
