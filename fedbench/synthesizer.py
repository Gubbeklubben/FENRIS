import numpy as np
from abc import ABC, abstractmethod

from fedbench.strategy import Strategy


class Synthesizer(ABC):
    def __init_subclass__(cls, **kwargs):
        # TODO: Register concrete impl.
        # Skip if cls is abstract.
        # Raise exc if already registered.
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_state: dict[str, np.ndarray]) -> Synthesizer:
        pass

    @abstractmethod
    @property
    def server_strategy(self) -> Strategy:
        pass

    # TODO! Figure out signature...
    @abstractmethod
    def local_train(self):
        pass

    # TODO! Figure out signature...
    @abstractmethod
    def sample(self):
        pass

