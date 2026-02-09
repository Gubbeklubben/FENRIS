import fedbench.algorithms as algorithms
from fedbench.algorithms import Algorithm


def test_builtins_produce_algorithms() -> None:
    for name, _ in algorithms.builtins():
        factory = algorithms.load_factory(name)
        instance = factory()
        assert isinstance(instance, Algorithm)