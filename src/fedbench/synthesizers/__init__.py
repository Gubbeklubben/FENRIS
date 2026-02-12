import importlib
import keyword
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points

from fedbench.synthesizers.synthesizer import ServerComponent

# Registry for builtin synthesizers. Manually maintained, and maps name
# to locator, in the interest of lazy imports.
_builtins = {
    "fed_smoke": f"{__package__}.fed_smoke:FedSmoke",
}

# Let users develop their synthesizers in independent packages
# without tinkering around inside the framework core. Note that plugins
# are fully trusted code.
_plugins = entry_points(group=__package__)


@dataclass(frozen=True)
class AlgorithmFactory:
    name: str
    locator: str
    factory: Callable[..., ServerComponent]

    def __call__(self, *args, **kwargs) -> ServerComponent:
        return self.factory(*args, **kwargs)


def builtins():
    yield from _builtins.items()


def plugins():
    for e in _plugins:
        yield e.name, e.value


def load_factory(name: str) -> AlgorithmFactory:
    factory = _load_builtin(name)

    if factory is None:
        factory = _load_plugin(name)

    if factory is None:
        raise ValueError(f"No such algorithm '{name}'")

    if not callable(factory.factory):
        raise TypeError(f"'{factory.locator}' is not callable")

    return factory


def _load_builtin(name: str) -> AlgorithmFactory | None:
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

    return AlgorithmFactory(
        name,
        locator,
        getattr(module, attr))


def _load_plugin(name: str) -> AlgorithmFactory | None:
    try:
        entry_point = _plugins[name]
    except KeyError:
        return None

    return AlgorithmFactory(
        name,
        entry_point.value,
        entry_point.load())


def _is_valid_locator(locator: str) -> bool:
    module, _, attr = locator.partition(":")
    def valid(s):
        return s.isidentifier() and not keyword.iskeyword(s)
    return all(valid(m) for m in module.split(".")) and valid(attr)