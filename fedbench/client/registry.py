from collections.abc import Callable

from fedbench._registry import ClientRegistry as _ClientRegistry
from fedbench.client.synthesizer import Synthesizer


_SYNTHESIZER_FACTORY = "_synthesizer_factory"


class ClientRegistry(_ClientRegistry):
    def synthesizer(
            self,
            factory: Callable[[], Synthesizer]) -> Callable[[], Synthesizer]:
        """Register a synthesizer factory."""
        return self._register(
            decorator_name="synthesizer",
            attr_name=_SYNTHESIZER_FACTORY,
            plugin=factory)

    def get_synthesizer_factory(self) -> Callable[[], Synthesizer] | None:
        return getattr(self, _SYNTHESIZER_FACTORY, None)