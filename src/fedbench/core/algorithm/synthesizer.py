from abc import ABC, abstractmethod

from pandas import DataFrame

from fedbench.core.payload import Payload


class Synthesizer(ABC):
    """The framework view of the model to train and sample from."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        """Attach globally computed preprocessing artifacts.

        Override if you depend on global_init to do preprocessing. Always
        called right after creating an instance.
        """

        pass

    def attach_client_cache(self, cache: Payload) -> None:
        """Attach client cache.

        Override if you need to a place to keep client local state beyond a
        single request. Always called right after attach_global_init_artifacts.
        """

        pass

    @abstractmethod
    def train(
        self,
        request: Payload,
        data: DataFrame,
    ) -> Payload:
        pass

    @abstractmethod
    def sample(
        self,
        request: Payload,
        num_rows: int,
        seed: int,
    ) -> DataFrame:
        pass
