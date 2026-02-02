import importlib
from collections.abc import Iterable
from importlib.metadata import entry_points


importlib.invalidate_caches()
ALGORITHM_ENTRY_POINTS = entry_points(group="fedbench.algorithms")


def algorithms() -> Iterable[str]:
    for entry_point in ALGORITHM_ENTRY_POINTS:
        yield entry_point.name


def load_algorithm(name: str):
    return ALGORITHM_ENTRY_POINTS[name].load()

