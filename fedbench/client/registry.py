from typing import Callable

from fedbench._registry import PluginRegistry
from fedbench.client.synthesizer import Synthesizer


# python >= 3.12
type SynthesizerFactory = Callable[[], Synthesizer]

_SYNTHESIZER_KEY = "synthesizer"


class ClientRegistry(PluginRegistry):
    def synthesizer(self, factory: SynthesizerFactory) -> SynthesizerFactory:
        """Register a synthesizer factory."""
        return self._register(_SYNTHESIZER_KEY, factory)

    # Internal ComponentResolver api
    @property
    def _synthesizer_factory(self):
        return self._get(_SYNTHESIZER_KEY)