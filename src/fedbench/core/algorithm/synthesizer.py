import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

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
    coordinator: Payload | None = None
    synthesizer: Payload | None = None


class Synthesizer(Component):
    """The framework view of the model to train and sample from."""

    SUPPORTS_COORDINATORS: set[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "SUPPORTS_COORDINATORS"):
            raise TypeError(
                f"{cls.__name__} must define the class attribute "
                f"SUPPORTS_COORDINATORS: set[str]"
            )
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
        pass

    @abstractmethod
    def global_init(
        self,
        dataset: DataFrame,
        context: GlobalInitContext,
    ) -> GlobalInitArtifacts:
        pass

    @abstractmethod
    def train(
        self,
        request: Payload,
        data: DataFrame,
        context: TrainContext,
    ) -> Payload:
        pass

    @abstractmethod
    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> DataFrame:
        pass
