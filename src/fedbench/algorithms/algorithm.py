from abc import ABC, abstractmethod
from collections.abc import Iterable

from fedbench.common import InitResponse, TrainResponse, MLRuntime, ModelState
from fedbench.algorithms.synthesizer import Synthesizer


# Despite the suggestion in the technical description I maintain for now
# that a slightly clearer server/client separation in the algorithm interface
# as suggested below is preferable.
# I see no reason not to let an algorithm instance live for an entire
# federation loop. It just lets implementations keep state in the instance,
# rather than finding other ways. Client side objects (synthesizers) on the
# other hand will die and be reborn each iteration.
class Algorithm(ABC):
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def server_ml_runtime(self) -> MLRuntime:
        pass

    @abstractmethod
    def server_initialize(
            self,
            responses: Iterable[InitResponse]) -> ModelState:
        pass

    @abstractmethod
    def server_aggregate(
            self,
            server_round: int,
            results: Iterable[TrainResponse]
    ) -> tuple[ModelState | None, dict[str, float] | None]:
        pass

    @abstractmethod
    def synthesizer_factory(self) -> Synthesizer:
        pass