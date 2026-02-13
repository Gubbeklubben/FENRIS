import importlib
import keyword
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points

from fedbench.algorithms.algorithm import (
    Algorithm,
    Aggregator,
    Synthesizer,
)

# Registry for builtin algorithms. Manually maintained, and maps name
# to locator, in the interest of lazy imports.
_builtins = {
    "fed_smoke": f"{__package__}.fed_smoke:FedSmoke",
}

# Let users develop their algorithms in independent packages
# without tinkering around inside the framework core. Note that plugins
# are fully trusted code.
_plugins = entry_points(group=__package__)


@dataclass(frozen=True)
class RegisteredAlgorithm:
    name: str
    locator: str
    cls: type[Algorithm]


def builtins() -> Iterable[tuple[str, str]]:
    yield from _builtins.items()


def plugins() -> Iterable[tuple[str, str]]:
    for e in _plugins:
        yield e.name, e.value


def load_algorithm(name: str) -> RegisteredAlgorithm:
    registered_alg = _load_builtin(name)

    if registered_alg is None:
        registered_alg = _load_plugin(name)

    if registered_alg is None:
        raise ValueError(f"No such algorithm '{name}'")

    return registered_alg


def _load_builtin(name: str) -> RegisteredAlgorithm | None:
    try:
        locator = _builtins[name]
    except KeyError:
        return None

    if not _is_valid_locator(locator):
        raise ValueError(
            f"Invalid locator '{locator}' in builtin algorithm registry")

    module_name, _, attr = locator.partition(":")
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise ValueError(
            f"Bad locator '{locator}' in builtin algorithm registry")

    return RegisteredAlgorithm(
        name,
        locator,
        getattr(module, attr))


def _load_plugin(name: str) -> RegisteredAlgorithm | None:
    try:
        entry_point = _plugins[name]
    except KeyError:
        return None

    return RegisteredAlgorithm(
        name,
        entry_point.value,
        entry_point.load())


def _is_valid_locator(locator: str) -> bool:
    module, _, attr = locator.partition(":")
    def valid(s):
        return s.isidentifier() and not keyword.iskeyword(s)
    return all(valid(m) for m in module.split(".")) and valid(attr)