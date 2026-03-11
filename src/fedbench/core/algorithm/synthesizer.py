from abc import ABC, abstractmethod

from pandas import DataFrame

from fedbench.core.data import TableSchema
from fedbench.core.update import Update


class Synthesizer(ABC):
    """The framework view of the model to train and sample from."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def attach_global_init_artifacts(self, artifacts: Update) -> None:
        """Attach globally computed preprocessing artifacts.

        Override if you depend on global_init to do preprocessing. Always
        called right after creating an instance.
        """

        pass

    def attach_client_cache(self, cache: Update) -> None:
        """Attach client cache.

        Override if you need to a place to keep client local state beyond a
        single request. Always called right after attach_global_init_artifacts.
        """

        pass

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def fed_init(
        self,
        request: Update,
        seed: int,
        schema: TableSchema,
        data: DataFrame,
    ) -> Update:
        return Update()

    @abstractmethod
    def train(
        self,
        request: Update,
        data: DataFrame,
    ) -> Update:
        pass

    @abstractmethod
    def sample(
        self,
        request: Update,
        num_rows: int,
        seed: int,
    ) -> DataFrame:
        pass
