from collections.abc import Iterable
from importlib.metadata import entry_points


_ALGORITHM_ENTRY_POINTS = entry_points(group="fedbench.algorithms")


def algorithms() -> Iterable[str]:
    for entry_point in _ALGORITHM_ENTRY_POINTS:
        yield entry_point.name


def load_algorithm(name: str):
    return _ALGORITHM_ENTRY_POINTS[name].load()

