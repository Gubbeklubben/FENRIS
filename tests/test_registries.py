from fedbench.algorithms import (
    registry as algorithm_reg,
    Algorithm,
    Synthesizer,
    Aggregator
)


def test_registered_algorithms_produce_expected_types() -> None:
    for name in algorithm_reg:
        algorithm = algorithm_reg.call(name)
        assert isinstance(algorithm, Algorithm)

        synthesizer = algorithm.create_synthesizer()
        assert isinstance(synthesizer, Synthesizer)

        aggregator = algorithm.create_aggregator()
        assert isinstance(aggregator, Aggregator)