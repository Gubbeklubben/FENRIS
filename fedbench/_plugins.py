from collections.abc import Iterable
from importlib.metadata import entry_points


_ALGORITHM_ENTRY_POINTS = entry_points(group="fedbench.algorithms")


def algorithms() -> Iterable[str]:
    for entry_point in _ALGORITHM_ENTRY_POINTS:
        yield entry_point.name


def load_algorithm(name: str):
    factory =  _ALGORITHM_ENTRY_POINTS[name].load()
    return factory()


def load_server_policy_factory(algorithm_name: str):
    algorithm = load_algorithm(algorithm_name)
    return algorithm.server_policy_factory


def load_synthesizer_factory(algorithm_name: str):
    algorithm = load_algorithm(algorithm_name)
    return algorithm.synthesizer_factory

