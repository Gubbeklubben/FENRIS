import fedbench.synthesizers as synthesizers
from fedbench.synthesizers.synthesizer import Synthesizer


def test_builtins_produce_synthesizers() -> None:
    for name, _ in synthesizers.builtins():
        factory = synthesizers.load_factory(name)
        instance = factory()
        assert isinstance(instance, Synthesizer), "Not a Synthesizer instance"