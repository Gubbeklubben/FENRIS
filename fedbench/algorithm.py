from abc import ABC, abstractmethod

from fedbench.server_policy import BaseServerPolicy
from fedbench.synthesizer import Synthesizer


class Algorithm(ABC):
    @abstractmethod
    def server_policy_factory(self) -> BaseServerPolicy:
        pass

    @abstractmethod
    def synthesizer_factory(self) -> Synthesizer:
        pass