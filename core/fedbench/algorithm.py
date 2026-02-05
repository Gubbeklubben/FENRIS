from abc import ABC, abstractmethod

from fedbench.server_policy import BaseServerPolicy
from fedbench.synthesizer import Synthesizer


class Algorithm(ABC):
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def server_policy_factory(self) -> BaseServerPolicy:
        pass

    @abstractmethod
    def synthesizer_factory(self) -> Synthesizer:
        pass