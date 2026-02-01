import importlib
import keyword

from fedbench._registry import BaseRegistry


def validate_locator(locator: str) -> bool:
    module, _, attr = locator.partition(":")
    def valid(s):
        return s.isidentifier() and not keyword.iskeyword(s)
    return all(valid(m) for m in module.split(".")) and valid(attr)


def load_registry(locator: str) -> BaseRegistry | None:
    module_name, _, attr = locator.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr, None)