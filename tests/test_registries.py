from fedbench.algorithms import (
    registry as alg_registry,
    Algorithm,
    Synthesizer,
    Aggregator
)


def test_registered_algorithms_produce_expected_types() -> None:
    for metadata in alg_registry:
        algorithm = alg_registry.load(metadata.name)
        assert issubclass(algorithm, Algorithm)

        synthesizer = algorithm.create_synthesizer()
        assert isinstance(synthesizer, Synthesizer)

        aggregator = algorithm.create_aggregator()
        assert isinstance(aggregator, Aggregator)