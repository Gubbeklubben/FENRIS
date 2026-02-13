from fedbench.algorithms import Algorithm, builtins, load_algorithm, \
    Synthesizer, Aggregator


def test_builtin_factories() -> None:
    for name, _ in builtins():
        algorithm_meta = load_algorithm(name)
        algorithm = algorithm_meta.cls
        assert issubclass(algorithm, Algorithm)

        synthesizer = algorithm.synthesizer_factory()
        assert isinstance(synthesizer, Synthesizer)

        aggregator = algorithm.aggregator_factory()
        assert isinstance(aggregator, Aggregator)